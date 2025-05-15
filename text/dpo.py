import argparse

from torch.utils.data import Dataset
from trl import DPOTrainer, ScriptArguments, DPOConfig, ModelConfig, TrlParser

import torch
import datasets
from tqdm import tqdm
import transformers
from transformers import set_seed
from datasets import disable_caching
from dataclasses import dataclass, field
from typing import Dict
import warnings

warnings.filterwarnings("ignore")

disable_caching()
set_seed(1234)
tqdm.pandas()


@dataclass
class TrainingArguments(ScriptArguments):
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    train_test_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    truncate_source: bool = field(default=False)


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def make_data(
        tokenizer: transformers.PreTrainedTokenizer, training_args: TrainingArguments
) -> Dict:
    """Make dataset and collator for pretrain."""
    rank0_print("Loading data...")

    def process_sample(sample):
        system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample["prompt"]}
        ]
        example = {
            'prompt': tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            'chosen': sample["chosen"],
            'rejected': sample["rejected"],
        }
        return example

    dataset = datasets.load_dataset(training_args.dataset_name, split="train").map(
        process_sample,
        load_from_cache_file=True,
        desc="Applying chat template into training data"
    )

    return dict(train_dataset=dataset, eval_dataset=None)


def train():
    global local_rank
    parser = TrlParser((TrainingArguments, DPOConfig, ModelConfig))
    training_args, dpo_args, model_args = parser.parse_args_and_config()
    args = {**training_args.__dict__, **dpo_args.__dict__, **model_args.__dict__}
    args = argparse.Namespace(**args)
    dpo_args.max_length = training_args.model_max_length

    local_rank = dpo_args.local_rank

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    ref_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=args.model_max_length
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    data = make_data(tokenizer, training_args)

    trainer = DPOTrainer(
        model,
        ref_model=ref_model,
        processing_class=tokenizer,
        args=dpo_args,
        **data
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)


if __name__ == "__main__":
    train()
