from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import load_dataset_builder, load_dataset
from torch.utils.data import IterableDataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from transformers.integrations import deepspeed

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled

    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


class TextDataset(IterableDataset):
    def __init__(self, data_path, tokenizer: transformers.PreTrainedTokenizer):
        super(TextDataset, self).__init__()
        self.tokenizer = tokenizer
        self.datasets = self._create_dataset(data_path)

    def _create_dataset(self, data_path):
        builder = load_dataset_builder(data_path)
        subsets = [config.name for config in builder.BUILDER_CONFIGS]
        print("dateset:", builder.dataset_name)
        print("\ndescription:", builder.info.description)
        print("\nsubsets:", subsets)
        print("\nsplits:", builder.info.splits)
        print("\nfeatures:", builder.info.features)

        return [load_dataset(data_path, subset, streaming=True) for subset in subsets] if subsets else [
            load_dataset(data_path, streaming=True)]

    def __iter__(self):
        for dataset in self.datasets:
            for split_name, split_dataset in dataset.items():
                print(f"process split: {split_name}")
                for sample in split_dataset:
                    input_ids = self.tokenizer(f'{sample["text"]}', truncation=True, padding="max_length",
                                               max_length=self.tokenizer.model_max_length).input_ids
                    targets = input_ids[1:] + [self.tokenizer.pad_token_id]
                    input_ids = torch.tensor(input_ids, dtype=torch.int)
                    targets = torch.tensor(targets, dtype=torch.int)
                    yield dict(
                        input_ids=input_ids,
                        labels=targets,
                        attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
                    )


def make_data(
        tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments
) -> Dict:
    """Make dataset and collator for pretrain."""
    rank0_print("Loading data...")

    train_dataset = TextDataset(data_args.data_path, tokenizer=tokenizer)

    if data_args.eval_data_path:
        eval_dataset = TextDataset(data_args.data_path, tokenizer=tokenizer)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    (
        model_args,
        data_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    model = transformers.AutoModelForCausalLM.from_config(config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load data
    data = make_data(tokenizer=tokenizer, data_args=data_args)

    # Start trainner
    trainer = Trainer(
        model=model, args=training_args, **data, processing_class=tokenizer
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
