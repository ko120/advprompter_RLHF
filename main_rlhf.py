# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from copy import copy
from datetime import datetime
from accelerate import  Accelerator
from peft import LoraConfig, PeftModel, get_peft_model
from datasets import load_dataset

import csv
import copy
import os

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import setproctitle
import torch
import re
from omegaconf import DictConfig, OmegaConf
from torchrl.data import ListStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from tqdm import tqdm
from collections import defaultdict

import wandb
from llm import LLM
from sequence import MergedSeq, Seq, collate_fn
import warnings
from utils import (
    Metrics,
    check_jailbroken,
    column_names,
    dotdict,
    get_dataloader,
    log_data,
    read_csv_file,
    hit_rate_at_n,
    AdvPromptDataset,
    llm_loader,
)
from custom_trl.trl.trainer import PPOConfig, PPOTrainer

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoConfig)
from advprompteropt import advPrompterOpt, evaluate_prompt
import pdb
from copy import deepcopy


setproctitle.setproctitle("model_train")

class Workspace:
    def __init__(self, cfg):
        pl.seed_everything(cfg.seed)
        self.step = 0
        self.cfg = cfg
        self.verbose = cfg.verbose
        self.enable_wandb = cfg.wandb_params.enable_wandb
        self.enable_accelerator = cfg.train.enable_accelerator
        self.starttime = datetime.now()

        if self.enable_wandb:
            self.init_wandb()
        tqdm.write("Initializing Prompter...")
        self.prompter = LLM(cfg.prompter, verbose=self.verbose)
        tqdm.write("Initializing TargetLLM...")
        self.target_llm = LLM(cfg.target_llm, verbose=self.verbose)
        tqdm.write("Initializing RewardLLM...")
        self.reward_llm = LLM(cfg.reward, verbose=self.verbose)
        tqdm.write("Initializing ValueLLM...")
        self.value_llm = LLM(cfg.value, verbose=self.verbose)
     
        # quantization_config = BitsAndBytesConfig(
        #         load_in_4bit=True,
        #         bnb_4bit_compute_dtype=torch.bfloat16,  # For consistency with model weights, we use the same value as `torch_dtype`
        #         bnb_4bit_quant_type="nf4",
        #         bnb_4bit_use_double_quant= True,
        #         bnb_4bit_quant_storage=torch.bfloat16, # we need to use bfloat16 instead of float16 for better numeric stability
        #     )

        
        # # if using RichardErkhov/cais_-_HarmBench-Mistral-7b-val-cls-4bits model is quantized on 32 bit, so input should be 32 bit 
        # self.reward_llm = AutoModelForSequenceClassification.from_pretrained("RichardErkhov/cais_-_HarmBench-Llama-2-13b-cls-4bits", 
        #                                                                      torch_dtype=torch.bfloat16,num_labels=1,
        #                                                                         quantization_config= quantization_config,
        #                                                                      )
        # self.value_llm = AutoModelForSequenceClassification.from_pretrained("RichardErkhov/cais_-_HarmBench-Llama-2-13b-cls-4bits", 
        #                                                                      torch_dtype=torch.bfloat16,num_labels=1,
        #                                                                      quantization_config = quantization_config,
        #                                                                      )
        # lora_config_dct = dict(self.cfg.value.llm_params.lora_params.lora_config)
        # lora_config_dct["target_modules"] = [
        #     m for m in self.cfg.value.llm_params.lora_params.lora_config["target_modules"]
        # ]
        # lora_config = LoraConfig(**lora_config_dct)
        # self.value_llm  = get_peft_model(self.value_llm , lora_config)
        # pdb.set_trace()
        self.test_prefixes = read_csv_file(self.cfg.data.test_prefixes_pth)
        self.affirmative_prefixes = read_csv_file(
            self.cfg.data.affirmative_prefixes_pth
        )

       
        
        self.train_loader = get_dataloader(
            data_pth=self.cfg.train.dataset_pth,
            shuffle=True,
            augment_target=self.cfg.train.augment_target,
            batch_size=self.cfg.train.batch_size,
        )
   
        self.eval_loader = get_dataloader(
            data_pth="/home/brianko/advprompter_RLHF/data/harmful_behaviors/dataset/validation.csv",
            shuffle=False,
            augment_target=True,
            batch_size=self.cfg.eval.batch_size,
        )
        
        # self.total_train_steps = self.cfg.train.epochs * self.train_loader.effective_dataset_size
        ppo_config_kwargs = dict(self.cfg.train.ppo_params.config)
        
        # trl==0.12.1
        self.ppo_config = PPOConfig(
            # num_ppo_epochs = self.total_train_steps,
            **ppo_config_kwargs
        )
        self.ppo_trainer = PPOTrainer(
            args=self.ppo_config,
            cfg= self.cfg,
            processing_class = self.prompter.tokenizer,
            model=self.prompter,
            ref_model=None,
            value_model = self.value_llm,
            reward_model=self.reward_llm,
            target_llm = self.target_llm,
            train_loader= self.train_loader,
            eval_loader = self.eval_loader,
            )
        
        self.train_table = wandb.Table(columns=column_names)
        self.eval_table = wandb.Table(columns=column_names)



    @torch.no_grad()
    def init_wandb(self):
        tqdm.write("Initializing Wandb...")
        wandb_id = wandb.util.generate_id()
        config = omegaconf.OmegaConf.to_container(
            self.cfg, resolve=True, throw_on_missing=True
        )
        wandb.init(
            entity=self.cfg.wandb_params.entity,
            project=self.cfg.wandb_params.project,
            config=config,
            id=wandb_id,
            resume="allow",
        )

    @torch.no_grad()
    def save_prompter(self):
        save_path = os.path.join(self.cfg.train.model_save_dir, f"step_{self.step}")
        tqdm.write(f" Saving prompter to {save_path}...")
        self.prompter.save_pretrained(save_path=save_path)

    def pretrain(self):
        tqdm.write("Starting pretraining...")
        pbar = tqdm(range(self.cfg.pretrain.epochs))
        pbar.set_description("Warmstarting (epochs)")
        for pretrain_epoch in pbar:
            self.pretrain_epoch()
        if self.cfg.pretrain.do_eval_after:
            self.eval()

    def pretrain_epoch(self):
        self.prompter.train()
        self.target_llm.eval()

        pretrain_metrics = Metrics(prefix="pretrain/")
        pretrain_loader = get_dataloader(
        data_pth=self.cfg.pretrain.dataset_pth,
        shuffle=True,
        augment_target=False,
        batch_size=self.cfg.pretrain.batch_size,
        
        )
        pbar_batches = tqdm(pretrain_loader)
        pbar_batches.set_description(f"Pre-training epoch {self.epoch}")
        for batch_idx, batch in enumerate(pbar_batches):
            context = self.batch_to_context(batch)
            instruct = context.instruct
            suffix = context.suffix
            prompter_tf_opt = self.finetune_prompter_step(
                instruct=instruct, suffix=suffix
            )
            log_data(
                log_table=self.train_table,
                metrics=pretrain_metrics,
                step=self.step,
                split=self.cfg.pretrain.dataset_key,
                batch_idx=batch_idx,
                test_prefixes=self.test_prefixes,
                affirmative_prefixes=self.affirmative_prefixes,
                batch_size=self.cfg.pretrain.batch_size,
                log_sequences_to_wandb=False,
                log_metrics_to_wandb=self.enable_wandb,
                prompter_tf_opt=prompter_tf_opt,
            )
            self.step += instruct.bs

        if self.enable_wandb:
            wandb.log(dict(train_examples=copy(self.train_table)), step=self.step)
        avg_metrics = pretrain_metrics.get_avg(
            step=self.step, log_to_wandb=self.enable_wandb
        )
        tqdm.write(
            f" Pretrain epoch opt loss: {avg_metrics['avg/pretrain/prompter/tf/opt/loss']:.2f}"
        )

    def train(self):
        self.ppo_trainer.train()

        if self.cfg.train.model_save_dir is not None:
            self.save_prompter()
        self.eval()

    

    def batch_to_context(self, batch):
        model_map = dict(
            instruct=self.prompter,
            suffix=self.prompter,
            target=self.target_llm,
            full_instruct=self.target_llm,
        )
        context = dotdict()
        for key, model in model_map.items():
            if key in batch.keys():
                seq = Seq(
                    text=batch[key],
                    tokenizer=model.tokenizer,
                    device=model.device,
                )
            else:
                seq = None
            context[key] = seq
        return context

    def add_to_replay_buffer(
        self,
        instruct,
        suffix,
        target,
        target_llm_tf,
        target_llm_tf_opt,
        target_llm_ar_opt,
    ):
        loss_batch = target_llm_tf.loss_batch
        loss_opt_batch = target_llm_tf_opt.loss_batch
        # priority = priority_factor.loss_delta * relu(loss_delta) + priority_factor.jailbreaking * jailbreaking
        priority = (
            torch.relu(loss_batch - loss_opt_batch)
            * self.cfg.train.replay_buffer.priority_factor.loss_delta
        )
        if self.cfg.train.replay_buffer.priority_factor.jailbreaking > 0:
            _, target_llm_ar_opt_jailbroken_list = check_jailbroken(
                seq=target_llm_ar_opt.response_sample,
                test_prefixes=self.test_prefixes,
            )
            jailbroken = torch.tensor(
                target_llm_ar_opt_jailbroken_list, device=loss_batch.device
            )
            priority += (
                jailbroken * self.cfg.train.replay_buffer.priority_factor.jailbreaking
            )
        for i, prio in enumerate(priority):
            if prio > 0:
                datapoint = (
                    instruct[i],
                    target[i],
                    suffix[i],
                    priority[i],
                )
                idx = self.replay_buffer.add(datapoint)
                self.replay_buffer.update_priority(index=idx, priority=prio.item())

    def finetune_prompter(self):
        prompter_tf_opt = None
        if len(self.replay_buffer) < self.cfg.train.batch_size:
            return None

        if self.verbose:
            tqdm.write(
                f"Step: {self.step} | Sampling from replay buffer and finetuning prompter..."
            )
        num_updates = min(
            self.cfg.train.replay_buffer.num_updates,
            len(self.replay_buffer) // self.cfg.train.batch_size,
        )
        for _ in range(num_updates):
            context, priority_batch = self.replay_buffer.sample(
                batch_size=self.cfg.train.batch_size
            )
            prompter_tf_opt = self.finetune_prompter_step(
                instruct=context.instruct, suffix=context.suffix
            )
            if self.verbose:
                tqdm.write(
                    f"Step: {self.step} | Regressing Prompter to sampled target suffixes: Loss {prompter_tf_opt.loss:.3f}, Sample priorities {[p.item() for p in priority_batch]}"
                )
        return prompter_tf_opt

    def finetune_prompter_step(self, instruct, suffix):
        self.prompter_optimizer.zero_grad()
        prompter_tf_opt = self.prompter.compute_pred_loss_teacher_forced(
            key="suffix",
            instruct=instruct,
            suffix=suffix,
            loss_params=dict(hard_labels=True),
        )
        loss = prompter_tf_opt.loss
        loss.backward()
        self.prompter_optimizer.step()
        if self.enable_wandb:
            wandb.log({"regression_loss": loss.item()}, step=self.step)
        return prompter_tf_opt

    @torch.no_grad()
    def eval(self):
        suffix_dataset_pth_dct = self.generate_suffix_datasets()
        self.eval_suffix_datasets(suffix_dataset_pth_dct)

    @torch.no_grad()
    def generate_suffix_datasets(self):
        suffix_dataset_pth_dct = {}
        for dataset_key, dataset_pth in self.cfg.eval.data.dataset_pth_dct.items():
            suffix_dataset = self.generate_suffix_dataset(
                dataset_key=dataset_key, dataset_pth=dataset_pth
            )
            suffix_dataset_pth = self.save_suffix_dataset(
                suffix_dataset, dir=self.cfg.eval.data.suffix_dataset_dir
            )
            suffix_dataset_pth_dct[suffix_dataset.suffix_dataset_key] = (
                suffix_dataset_pth
            )
        return suffix_dataset_pth_dct

    @torch.no_grad()
    def generate_suffix_dataset(self, dataset_key, dataset_pth):
        self.prompter.eval()
        self.target_llm.eval()

        if self.cfg.prompter.gen_params.do_sample:
            num_trials = self.cfg.eval.num_trials
        else:
            if self.cfg.eval.num_trials != 1:
                warnings.warn(
                    "Prompter generation is deterministic, but num_trials > 1. Setting num_trials to 1."
                )
            num_trials = 1

        data = []

        suffix_dataset_key = f"{dataset_key}_{self.step}"
        eval_loader = get_dataloader(
            data_pth=dataset_pth,
            shuffle=False,
            augment_target=False,
            batch_size=self.cfg.eval.batch_size,
        )
        pbar_batches = tqdm(eval_loader)
        pbar_batches.set_description(f"Generating suffix dataset {suffix_dataset_key}")
        for batch in pbar_batches:
            context = self.batch_to_context(batch)
            instruct = context.instruct
            target = context.target
            batch_data = []
            for max_new_tokens in self.cfg.eval.prompter.max_new_tokens_list:
                trial_data = []
                for trial in range(num_trials):
                    prompter_ar = self.prompter.generate_autoregressive(
                        key="suffix",
                        max_new_tokens=max_new_tokens,
                        instruct=instruct,
                    )
                    suffix = prompter_ar.response_sample

                    full_instruct = MergedSeq(seqs=[instruct, suffix]).to_seq(
                        merge_dtype="ids"
                    )

                    assert instruct.bs == target.bs == suffix.bs
                    datapoint = []
                    for i in range(instruct.bs):
                        datapoint.append(
                            (
                                instruct.text[i],
                                target.text[i],
                                suffix.text[i],
                                full_instruct.text[i],
                            )
                        )
                    trial_data.append(datapoint)
                batch_data.append(trial_data)

            for i in range(instruct.bs):
                for j in range(len(self.cfg.eval.prompter.max_new_tokens_list)):
                    for k in range(num_trials):
                        data.append(batch_data[j][k][i])

        suffix_dataset = dotdict(
            data=data,
            fields=["instruct", "target", "suffix", "full_instruct"],
            suffix_dataset_key=suffix_dataset_key,
        )

        return suffix_dataset

    @torch.no_grad()
    def save_suffix_dataset(self, suffix_dataset, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        suffix_dataset_pth = os.path.join(
            dir,
            suffix_dataset.suffix_dataset_key + ".csv",
        )
        tqdm.write(
            f" Saving {suffix_dataset.suffix_dataset_key} to {suffix_dataset_pth}"
        )
        with open(suffix_dataset_pth, "w") as csvfile:
            csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
            csvwriter.writerow(suffix_dataset.fields)
            csvwriter.writerows(suffix_dataset.data)
        return suffix_dataset_pth

    @torch.no_grad()
    def eval_suffix_datasets(self, suffix_dataset_pth_dct):
        for suffix_dataset_key, suffix_dataset_pth in suffix_dataset_pth_dct.items():
            self.eval_suffix_dataset(
                suffix_dataset_key=suffix_dataset_key,
                suffix_dataset_pth=suffix_dataset_pth,
            )

    @torch.no_grad()
    def eval_suffix_dataset(self, suffix_dataset_key, suffix_dataset_pth):
        self.prompter.eval()
        self.target_llm.eval()

        # split = suffix_dataset_key
        split = re.sub("[^a-zA-Z]", "", suffix_dataset_key)

        eval_loader = get_dataloader(
            suffix_dataset_pth,
            shuffle=False,
            augment_target=False,
            batch_size=self.cfg.eval.batch_size,
        )
        eval_metrics = Metrics(prefix=split + "_eval/")

        instruct_jb_dict = defaultdict(list)
        processed_samples, ppl_sum = 0, 0
        pbar = tqdm(eval_loader)
        pbar.set_description(
            f"Evaluating suffix dataset {suffix_dataset_key} | Jailbroken 0/0 | Success 0/0"
        )
        for batch_idx, batch in enumerate(pbar):
            context = self.batch_to_context(batch)
            instruct = context.instruct
            suffix = context.suffix
            full_instruct = context.full_instruct
            target = context.target
            target_llm_tf, target_llm_ar, basemodel_tf = evaluate_prompt(
                cfg=self.cfg,
                instruct=instruct,
                suffix=suffix,
                full_instruct=full_instruct,
                target=target,
                prompter=self.prompter,
                target_llm=self.target_llm,
                generate_target_llm_response=True,
            )

            # --------- check jb for each trial
            _, jailbroken_list = check_jailbroken(
                seq=target_llm_ar.response_sample, test_prefixes=self.test_prefixes
            )
            instruct = instruct
            assert instruct.bs == len(jailbroken_list)
            instruct_text = instruct.text
            for i in range(instruct.bs):
                instruct_jb_dict[instruct_text[i]].append(jailbroken_list[i])
            # -----------

            log_data(
                log_table=None,
                metrics=eval_metrics,
                step=self.step,
                split=split,
                batch_idx=batch_idx,
                test_prefixes=self.test_prefixes,
                affirmative_prefixes=self.affirmative_prefixes,
                batch_size=self.cfg.eval.batch_size,
                log_sequences_to_wandb=False,
                log_metrics_to_wandb=False,
                target_llm_tf=target_llm_tf,
                target_llm_ar=target_llm_ar,
                basemodel_tf=basemodel_tf,
            )
            processed_samples += instruct.bs
            if basemodel_tf is not None:
                ppl_sum += basemodel_tf.perplexity.sum().item()

            total_jailbroken = sum(
                eval_metrics.metrics[split + "_eval/target_llm/ar/jailbroken_sum"]
            )
            pbar.set_description(
                f"Evaluating {suffix_dataset_key} | Jailbroken {total_jailbroken}/{processed_samples}"
            )

        avg_metrics = eval_metrics.get_avg(step=self.step, log_to_wandb=False)
        avg_metrics["avg/" + split + "_eval/target_llm/ar/jailbroken_sum"] = (
            float(
                sum(eval_metrics.metrics[split + "_eval/target_llm/ar/jailbroken_sum"])
            )
            / processed_samples
        )

        tqdm.write(
            f" Loss: {avg_metrics['avg/' + split + '_eval/target_llm/tf/loss']:.2f}"
        )
        tqdm.write(
            f" Jailbroken: {avg_metrics['avg/' + split + '_eval/target_llm/ar/jailbroken_sum']:.2f}"
        )
        tqdm.write(f" PPL: {float(ppl_sum) / processed_samples:.2f}")
        jb_all = [jb_list for (instruct, jb_list) in instruct_jb_dict.items()]
        max_length = max(len(sublist) for sublist in jb_all)
        padded_list = [
            np.pad(sublist, (0, max_length - len(sublist)), "constant")
            for sublist in jb_all
        ]
        jb_stat_np = np.array(padded_list)
        for ti in range(1, jb_stat_np.shape[1] + 1):
            tqdm.write(
                f"{suffix_dataset_key} | hit rate @ {ti}: {hit_rate_at_n(jb_stat_np, ti)}"
            )
        if self.enable_wandb:
            wandb.log(avg_metrics, step=self.step)
            wandb.log(dict(eval_examples=copy(self.eval_table)), step=self.step)


@hydra.main(version_base=None, config_path="conf")
def main(cfg: DictConfig):
    tqdm.write("Starting run...")
    tqdm.write(f"Using parameters: \n{OmegaConf.to_yaml(cfg)}")
    workspace = Workspace(cfg)
    if cfg.mode == "train":
        workspace.train()
    elif cfg.mode == "eval":
        workspace.eval()
    elif cfg.mode == "eval_suffix_dataset":
        workspace.eval_suffix_datasets(cfg.eval.suffix_dataset_pth_dct)
    else:
        raise ValueError(f"Mode {cfg.mode} not recognized.")
    tqdm.write("Finished!")


if __name__ == "__main__":
    main()
