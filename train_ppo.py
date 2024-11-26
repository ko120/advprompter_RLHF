import numpy as np
import torch
from utils import Metrics, read_csv_file, get_dataloader, log_gradient_statistics, \
    compute_scheduled_param, column_names, log_data, \
    dotdict, apply_repetition_penalty, dump_json_q_step, check_jailbroken, get_total_allocated_memory
import hydra
from omegaconf import DictConfig, OmegaConf
from llm import LLM
from sequence import Seq, collate_fn, MergedSeq, stack_seqs
# from replay_buffer import ReplayBuffer
import omegaconf
import wandb
import setproctitle
setproctitle.setproctitle('llm-attacks-train')
from copy import copy
from tqdm import tqdm
import pytorch_lightning as pl
import sys
from datetime import datetime
from pprint import pprint
import time
from trl import PPOConfig, PPOTrainer
from trl.core import LengthSampler
# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(
#     mode='Plain', color_scheme='Neutral', call_pdb=1)
    

class Workspace:
    def __init__(self, cfg):
        pl.seed_everything(cfg.seed)
        self.step = 0
        self.cfg = cfg
        self.verbose = cfg.verbose
        self.enable_wandb = cfg.wandb_params.enable_wandb
        self.starttime = datetime.now()
        self.cfg.eval.eval_every = max(int(cfg.train.epochs / 10), cfg.eval.eval_every)

        if self.enable_wandb:
            self.init_wandb()
        
        training_dataloader_train_split = get_dataloader(data_params=cfg.data.train, augment_target=self.cfg.train.augment_target)
        evaluation_dataloader_train_split = get_dataloader(data_params=cfg.data.train, augment_target=self.cfg.eval.augment_target)
        evaluation_dataloader_eval_split = get_dataloader(data_params=cfg.data.eval, augment_target=self.cfg.eval.augment_target)
        evaluation_dataloader_test_split = get_dataloader(data_params=cfg.data.test, augment_target=self.cfg.eval.augment_target)
        self.training_dataloaders = dict(train=training_dataloader_train_split)
        self.evaluation_dataloaders = dict(train=evaluation_dataloader_train_split, validation=evaluation_dataloader_eval_split, test=evaluation_dataloader_test_split)

        self.test_prefixes = read_csv_file(cfg.data.test_prefixes_pth)
        self.affirmative_prefixes = read_csv_file(cfg.data.affirmative_prefixes_pth)

        if not cfg.prompter.llm_params.for_ppo:
            raise ValueError("For_ppo in prompter should be set to True for train_ppo.py")
        self.prompter = LLM(cfg.prompter, verbose=self.verbose)
        self.target_llm = LLM(cfg.target_llm, verbose=self.verbose)
        if self.prompter.tokenizer.vocab_size != self.target_llm.tokenizer.vocab_size and not self.cfg.target_llm.mediate_via_text:
            raise Exception(f'Vocab sizes of prompter {self.prompter.tokenizer.vocab_size} and target {self.target_llm.tokenizer.vocab_size} do not match.'
                            f'Consider setting cfg.target_llm.mediate_via_text=True')
        if self.enable_wandb and cfg.wandb_params.watch_params is not None:
            wandb.watch(self.prompter.model, **self.cfg.wandb_params.watch_params)

        self.total_train_steps = self.cfg.train.epochs * self.training_dataloaders['train'].effective_dataset_size

        ppo_config_kwargs = dict(cfg.train.ppo_params.config)
        self.ppo_config = PPOConfig(
            model_name=cfg.prompter.llm_params.model_name,
            log_with="wandb" if self.enable_wandb else None,
            batch_size=cfg.data.train.dataloader.batch_size,
            optimize_cuda_cache=True,
            seed=cfg.seed,
            steps=self.total_train_steps,
            **ppo_config_kwargs
        )

        self.ppo_trainer = PPOTrainer(
            self.ppo_config,
            self.prompter.model,
            ref_model=None,
            tokenizer=self.prompter.tokenizer,
        )

        self.train_table = wandb.Table(columns=column_names)
        self.eval_table = wandb.Table(columns=column_names)

    def init_wandb(self):
        wandb_id = wandb.util.generate_id()
        config = omegaconf.OmegaConf.to_container(
            self.cfg, resolve=True, throw_on_missing=True
        )
        wandb.init(
            entity='llm-attack-gang',
            project='llm-attacks', config=config,
            id=wandb_id, resume='allow'
        )

    def train(self):
        self.suffix_length = self.cfg.train.ppo_params.gen_params.max_new_tokens

        if self.cfg.eval.do_initial_eval:
            self.eval(split='validation')
            if self.cfg.eval.also_evaluate_on_train:
                self.eval(split='train')

        for epoch in range(self.cfg.train.epochs):
            if self.cfg.train.dynamic_suffix_length:
                self.suffix_length = min(epoch + 1, self.cfg.train.ppo_params.gen_params.max_new_tokens)
            print(f"Epoch {epoch}: ")
            self.train_epoch(split='train')

            if self.cfg.eval.eval_every is not None and (epoch + 1) % self.cfg.eval.eval_every == 0 and (epoch + 1) < self.cfg.train.epochs:
                if self.cfg.eval.save_trained_model:
                    self.save_prompter()
                self.eval(split='validation')
                if self.cfg.eval.also_evaluate_on_train:
                    self.eval(split='train')

        print("Training completed.")
        self.suffix_length = self.cfg.train.q_params.max_new_tokens

        if self.cfg.eval.do_final_eval:
            if self.cfg.eval.save_trained_model:
                self.save_prompter()
            print("Evaluating the prompter on the validation data...", flush=True)
            self.eval(split='validation', report_hit_success_rates=True)
            print("Evaluating the prompter on the original train data...", flush=True)
            self.eval(split='train', report_hit_success_rates=True)

    def save_prompter(self):
        time = self.starttime.strftime("%Y%m%d-%H%M%S")
        if self.enable_wandb:
            name = f'{time}_{wandb.run.name}_{wandb.run.id}/step_{self.step}'
        else:
            name = f'{time}_unnamed/step_{self.step}'
        self.prompter.save_pretrained(name=name)

    @torch.no_grad()
    def evaluate_context(self, context, generate_target_llm_response, verbose):
        basemodel_tf = None
        if context.suffix is not None:
            if verbose:
                print('--- Suffix:')
                pprint(context.suffix.text)
            try:
                basemodel_tf = self.prompter.compute_pred_loss_teacher_forced(
                    key='suffix', context=context, use_basemodel=True, loss_params=dict(hard_labels=True, loss_type='ce'))
                if verbose:
                    print(f'Perplexity suffix: {basemodel_tf.perplexity}', flush=True)
            except Exception as e:
                print(f"!!!!!!!!!!! [Warning!] Basemodel forward throw an exception: {e}\n Skipping...")

        full_instruct_seq = MergedSeq(seqs=[context.instruct, context.suffix]).to_seq(merge_data_type='ids')
        if self.cfg.target_llm.mediate_via_text:
            context.full_instruct = full_instruct_seq.text
        else:
            context.full_instruct = full_instruct_seq
 
        target_llm_tf = self.target_llm.compute_pred_loss_teacher_forced(
            key='target', context=context, loss_params=dict(hard_labels=True, loss_type='ce', reweight_loss=self.cfg.train.reweight_loss))
        if verbose:
            print('--- Query:')
            pprint(target_llm_tf.query.text)
        if verbose:
            pprint(f'--- TF Loss: {target_llm_tf.loss:.3f}')
            print('--- TF Response:')
            pprint(target_llm_tf.response_dist.text)
            print('--- Target:')
            pprint(context.target.text)
            
        if generate_target_llm_response:
            target_llm_ar = self.target_llm.generate_autoregressive(key='target', context=context)
            if verbose:
                print('--- AR Response:')
                pprint(target_llm_ar.response_sample.text)
        else:
            target_llm_ar = None
        return target_llm_tf, target_llm_ar, basemodel_tf

    def empty_cuda_cache(self):
        print(f'Memory before: {get_total_allocated_memory()} GB')
        num_gpus = torch.cuda.device_count()
        for gpu_id in range(num_gpus):
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
        print(f'Memory after: {get_total_allocated_memory()} GB')

    def train_epoch(self, split):
        self.prompter.train()
        self.target_llm.eval()
        dataloader = self.training_dataloaders[split]
        train_metrics = Metrics(prefix='train/')

        start_step = self.step

        print(f" Starting train epoch ({split} split) @ {start_step}...", flush=True)
        max_loss = self.cfg.train.ppo_params.max_loss_const

        for batch_idx, context in enumerate(tqdm(dataloader)):
            instruct_str = context.instruct
            target_str = context.target
            context.instruct = Seq(text=instruct_str, tokenizer=self.prompter.tokenizer, device=self.prompter.device)
            context.target = Seq(text=target_str, tokenizer=self.target_llm.tokenizer, device=self.target_llm.device)

            # ------- Rollout -----------
            self.prompter.prompt_manager.set_prompt(context)
            query_seq = self.prompter.prompt_manager.get_merged_prompt(up_to_key='suffix')
            query_ids = query_seq.ids
            query_mask = query_seq.mask
            query_id_list = [ids[mask_i] for ids, mask_i in zip(query_ids, query_mask)]

            gen_params = dict(self.cfg.train.ppo_params.gen_params)
            gen_params['max_new_tokens'] = self.suffix_length
            # gen_params["pad_token_id"] = self.prompter.tokenizer.pad_token_id
            # print("pad_token_id: ", self.prompter.tokenizer.pad_token_id)
            generation_config = self.prompter.get_generation_config()
            gen_params['generation_config'] = generation_config
            prompter_responses = self.ppo_trainer.generate(query_id_list, return_prompt=False, **gen_params)

            # verify that self.prompter.generate_autoregressive == self.ppo_trainer.generate
            # with torch.no_grad():
            #     prompter_ar = self.prompter.generate_autoregressive(key='suffix', context=context, max_new_tokens=self.suffix_length)
            #     context.suffix = prompter_ar.response_sample
            # print("!!!!!! context.suffix.ids")
            # pprint(context.suffix.ids)
            # print("!!!!!! ppo_trainer.generate")
            # pprint(prompter_responses)
            # ----------------------------

            # ------- Computing Rewards -----------
            # combining query+suffix
            prompter_responses_tensor = self.prompter.tokenizer.pad({
                'input_ids': [t.tolist() for t in prompter_responses]
            }, padding='longest', return_tensors='pt')
            prompter_responses_tensor = prompter_responses_tensor['input_ids'].to(self.prompter.device)
            context.suffix = Seq(ids=prompter_responses_tensor, tokenizer=self.prompter.tokenizer, device=self.prompter.device)
            full_instruct_seq = MergedSeq(seqs=[context.instruct, context.suffix]).to_seq(merge_data_type='ids')
            if self.cfg.target_llm.mediate_via_text:
                context.full_instruct = full_instruct_seq.text
            else:
                context.full_instruct = full_instruct_seq

            target_llm_tf = self.target_llm.compute_pred_loss_teacher_forced(
                key='target', context=context,
                loss_params=dict(hard_labels=True, loss_type='ce', reweight_loss=self.cfg.train.reweight_loss))
            # transforming loss -> reward = 1- loss 
            rewards = [max_loss-li.clone().detach() for li in target_llm_tf.loss_batch.squeeze()]
            target_llm_ar = None
            # adding jb score if use_jb_as_score is set to True
            if self.cfg.train.ppo_params.use_jb_as_score:
                target_llm_ar = self.target_llm.generate_autoregressive(key='target', context=context, max_new_tokens=self.suffix_length)
                print("Response from TargetLLM:")
                pprint(target_llm_ar.response_sample.text)
                _, jailbroken_list = check_jailbroken(seq=target_llm_ar.response_sample, test_prefixes=self.test_prefixes)
                rewards = [torch.tensor(float(jr)) + rewards[ji] for ji, jr in enumerate(jailbroken_list)]
            pprint(f'--- TF-losses: {target_llm_tf.loss_batch.squeeze()}, Rewards: {rewards}')
            # ----------------------------

            # ------- Perform PPO step -----------
            stats = self.ppo_trainer.step(query_id_list, prompter_responses, rewards)
            batch_ppo = {"query_id_list": query_id_list, "prompter_response_ids": prompter_responses,
                         "response": self.prompter.tokenizer.batch_decode(prompter_responses, skip_special_tokens=True),
                         "query": self.prompter.tokenizer.batch_decode(query_id_list, skip_special_tokens=True)}
            self.ppo_trainer.log_stats(stats, batch_ppo, rewards)
            # ----------------------------

            log_sequences = batch_idx % self.cfg.wandb_params.log_sequences_every.train == 0
            log_data(batch_size=dataloader.batch_size, log_table=self.train_table, metrics=train_metrics, step=self.step, split=split,
                     batch_idx=batch_idx, test_prefixes=self.test_prefixes, affirmative_prefixes=self.affirmative_prefixes,
                     log_sequences_to_wandb=log_sequences and self.enable_wandb, log_metrics_to_wandb=self.enable_wandb,
                     target_llm_tf=target_llm_tf, target_llm_ar=target_llm_ar)
            
            if self.enable_wandb and not self.cfg.wandb_params.log_table_at_epoch_end_only:
                wandb.log(dict(train_examples=copy(self.train_table)), step=self.step)
            self.step += dataloader.batch_size
            print()

            # self.empty_cuda_cache()
        
        if self.enable_wandb and self.cfg.wandb_params.log_table_at_epoch_end_only:
            wandb.log(dict(train_examples=copy(self.train_table)), step=self.step)

        avg_metrics = train_metrics.get_avg(step=self.step, log_to_wandb=self.enable_wandb)
        print(f" Train loss ({split} split) @ ({start_step}-{self.step}): {avg_metrics['avg/train/target_llm/tf/loss']:.2f}")

    def eval(self, split, report_hit_success_rates=False):
        self.prompter.eval()
        self.target_llm.eval()
        dataloader = self.evaluation_dataloaders[split]
        eval_metrics = Metrics(prefix=split + '_eval/')

        print(f" Starting eval ({split} split) @ {self.step}...", flush=True)
        processed_samples = 0
        jb_mean_all = []
        for batch_idx, context in enumerate(tqdm(dataloader)):
            context.instruct = Seq(text=context.instruct, tokenizer=self.prompter.tokenizer, device=self.prompter.device)
            context.target = Seq(text=context.target, tokenizer=self.target_llm.tokenizer, device=self.target_llm.device)

            log_sequences_to_wandb = batch_idx % self.cfg.wandb_params.log_sequences_every.eval == 0 and self.enable_wandb
            bs = context.instruct.bs
            min_prompter_ar_response_ids = torch.ones((bs, max(self.cfg.eval.prompter.max_new_tokens_list)), device=self.prompter.device, dtype=torch.long) * self.prompter.tokenizer.pad_token_id
            min_loss_batch = torch.ones((bs,), device=self.target_llm.device) * sys.maxsize
            context.suffix = None
            num_trials = self.cfg.eval.max_eval_trials if self.cfg.prompter.gen_params.do_sample else 1
            jb_list_batch, trial_cnt = [0.0] * bs, 0
            for trial in range(num_trials):
                for max_new_tokens in self.cfg.eval.prompter.max_new_tokens_list:
                    try:
                        prompter_ar = self.prompter.generate_autoregressive(key='suffix', context=context, max_new_tokens=max_new_tokens)
                    except Exception as e:
                        print("!!!!!!!!!!! [Warning!] Error during evaluation (generate_autoregressive) for batch: \n", context)
                        print(f'Exception: {e}')
                        print("!!!!!! Skipping the trial......")
                        continue
                    context.suffix = prompter_ar.response_sample
                    target_llm_tf, target_llm_ar, basemodel_tf = self.evaluate_context(context,
                                                                                       generate_target_llm_response=report_hit_success_rates,
                                                                                       verbose=self.verbose)
                    loss_batch = target_llm_tf.loss_batch
                    if report_hit_success_rates:
                        # success rate and hit rate
                        _, jailbroken_list = check_jailbroken(seq=target_llm_ar.response_sample, test_prefixes=self.test_prefixes)
                        jb_list_batch = [jb_list_batch[jbi] + jailbroken_list[jbi] for jbi in range(bs)]
                        trial_cnt += 1.
                    
                    for i, (ids, loss) in enumerate(zip(prompter_ar.response_sample.ids, loss_batch)):
                        if loss < min_loss_batch[i]:
                            min_prompter_ar_response_ids[i, :max_new_tokens] = ids
                            min_prompter_ar_response_ids[i, max_new_tokens:] = self.prompter.tokenizer.pad_token_id
                            min_loss_batch[i] = loss
                    context.suffix = None

            if min_prompter_ar_response_ids is None:
                print("!!!!!!!!!!! [Warning!] Eval the batch went wrong. Skipping the batch......")
                continue
            min_prompter_ar = prompter_ar
            min_prompter_ar.response_sample.ids = min_prompter_ar_response_ids
            context.suffix = min_prompter_ar.response_sample
            target_llm_tf, target_llm_ar, basemodel_tf = self.evaluate_context(context, generate_target_llm_response=True, verbose=self.verbose)

            if report_hit_success_rates:
                jb_list_batch = [jb_i / trial_cnt for jb_i in jb_list_batch]
                # print("jb_list_batch: ", jb_list_batch)
                jb_mean_all += jb_list_batch

            log_data(log_table=self.eval_table, metrics=eval_metrics, step=self.step, split=split,
                     batch_idx=batch_idx, test_prefixes=self.test_prefixes, affirmative_prefixes=self.affirmative_prefixes, 
                     batch_size=dataloader.batch_size,
                     log_sequences_to_wandb=log_sequences_to_wandb, log_metrics_to_wandb=False,
                     prompter_ar=min_prompter_ar, target_llm_tf=target_llm_tf, 
                     target_llm_ar=target_llm_ar, basemodel_tf=basemodel_tf)
            # prompter_tf_basemodel=prompter_tf_basemodel)
            processed_samples += context.target.bs
            print("Processed so far:", processed_samples,
                  " | Total Jailbroken (one shot):", sum(eval_metrics.metrics[split + '_eval/target_llm/ar/jailbroken_sum']),
                  " | Total Success (one shot):", sum(eval_metrics.metrics[split + '_eval/target_llm/ar/success_sum']), flush=True)

        print("Total processed samples: ", processed_samples, flush=True)
        avg_metrics = eval_metrics.get_avg(step=self.step, log_to_wandb=False)
        avg_metrics['avg/' + split + '_eval/target_llm/ar/jailbroken_sum'] = float(
            sum(eval_metrics.metrics[split + '_eval/target_llm/ar/jailbroken_sum'])) / processed_samples
        avg_metrics['avg/' + split + '_eval/target_llm/ar/success_sum'] = float(
            sum(eval_metrics.metrics[split + '_eval/target_llm/ar/success_sum'])) / processed_samples
        print(f" Eval loss ({split} split) @ {self.step}: {avg_metrics['avg/' + split + '_eval/target_llm/tf/loss']:.4f}")
        print(f" Eval jailbroken ({split} split) @ {self.step}: {avg_metrics['avg/' + split + '_eval/target_llm/ar/jailbroken_sum']:.4f}")
        print(f" Eval success ({split} split) @ {self.step}: {avg_metrics['avg/' + split + '_eval/target_llm/ar/success_sum']:.4f}")
        if report_hit_success_rates:
            jb_mean_all = np.array(jb_mean_all)
            avg_metrics['avg/' + split + '_eval/target_llm/ar/hit_rate'] = np.where(jb_mean_all > 0, 1.0, jb_mean_all).mean()
            avg_metrics['avg/' + split + '_eval/target_llm/ar/success_rate'] = jb_mean_all.mean()
            print(f" Eval jailbreak hit rate ({split} split) @ {self.step}: {avg_metrics['avg/' + split + '_eval/target_llm/ar/hit_rate']:.4f}")
            print(f" Eval jailbreak success rate ({split} split) @ {self.step}: {avg_metrics['avg/' + split + '_eval/target_llm/ar/success_rate']:.4f}")
        print("-------------\n")
        if self.enable_wandb:
            wandb.log(avg_metrics, step=self.step)
            wandb.log(dict(eval_examples=copy(self.eval_table)), step=self.step)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig):
    print("Starting run...")
    print(f'Using parameters: \n{OmegaConf.to_yaml(cfg)}', flush=True)
    workspace = Workspace(cfg)
    workspace.train()
    print("Finished!")

        
if __name__ == '__main__':
    main()
