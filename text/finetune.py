from dataclasses import dataclass, field
import logging
import os
from typing import Dict, Optional, List
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
import transformers
from transformers import Trainer, GPTQConfig
from transformers.integrations import deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from accelerate.utils import DistributedType

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
    lazy_preprocess: bool = False


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
    use_lora: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["lm_head"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def preprocess(
        conversations,
        tokenizer: transformers.PreTrainedTokenizer,
        system_message: str = "You are a helpful assistant."
) -> Dict:
    roles = {"human": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.convert_tokens_to_ids(["<|im_start|>"])[0]
    im_end = tokenizer.convert_tokens_to_ids(["<|im_end|>"])[0]
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    if roles[conversations[0]["from"]] != roles["human"]:
        conversations = conversations[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(conversations):
        role = roles[sentence["from"]]
        _input_id = tokenizer(role).input_ids + nl_tokens + \
                    tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == '<|im_start|>user':
            _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == '<|im_start|>assistant':
            _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                      _input_id[len(tokenizer(role).input_ids) + 1:-2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target
    assert len(input_id) == len(target)
    input_id += [tokenizer.pad_token_id] * (tokenizer.model_max_length - len(input_id))
    target += [IGNORE_TOKEN_ID] * (tokenizer.model_max_length - len(target))
    input_ids = torch.tensor(input_id[:tokenizer.model_max_length], dtype=torch.int)
    targets = torch.tensor(target[:tokenizer.model_max_length], dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class ChatDataset(Dataset):

    def __init__(self, data_path, tokenizer: transformers.PreTrainedTokenizer):
        super(ChatDataset, self).__init__()
        self.tokenizer = tokenizer
        self.dataset = self._create_dataset(data_path)

    def _create_dataset(self, data_path):
        data_files = []
        if os.path.isdir(data_path):
            for root, _, files in os.walk(data_path):
                for file in files:
                    data_files.append(os.path.join(root, file))
        else:
            data_files.append(data_path)

        return load_dataset("json", data_files=data_files)

    def __getitem__(self, i):
        return preprocess(self.dataset['train'][i]['conversations'], self.tokenizer)

    def __len__(self):
        return self.dataset['train'].num_rows


def make_data(
        tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    rank0_print("Loading data...")

    train_dataset = ChatDataset(data_args.data_path, tokenizer)

    if data_args.eval_data_path:
        eval_dataset = ChatDataset(data_args.eval_data_path, tokenizer)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # This serves for single-gpu qlora.
    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1)) == 1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are incompatible with QLoRA."
            )
        os.environ['TOKENIZERS_PARALLELISM'] = "true"
        training_args.label_names = ["labels"]

    is_chat_model = 'chat' in model_args.model_name_or_path.lower()
    if (
            training_args.use_lora
            and not lora_args.q_lora
            and deepspeed.is_deepspeed_zero3_enabled()
            and not is_chat_model
    ):
        raise RuntimeError("ZeRO3 is incompatible with LoRA when finetuning on base model.")

    model_load_kwargs = {
        'low_cpu_mem_usage': not deepspeed.is_deepspeed_zero3_enabled(),
    }

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=GPTQConfig(
            bits=4, dataset="c4", use_exllama=False
        )
        if training_args.use_lora and lora_args.q_lora
        else None,
        **model_load_kwargs,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if training_args.use_lora:
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type=TaskType.CAUSAL_LM,
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)

        # Print peft trainable params
        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    # Load data
    data_module = make_data(
        tokenizer=tokenizer, data_args=data_args
    )

    # Start trainner
    trainer = Trainer(
        model=model, args=training_args, **data_module, processing_class=tokenizer
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)


if __name__ == "__main__":
    train()
