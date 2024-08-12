import os
import sys
from typing import List
import fire
from utils.prompter import Prompter
from nvitop import select_devices
import time
from dataclasses import dataclass,field
import torch
import transformers
from utils.BalanceTrainer import BalanceTrainer
from datasets import load_dataset, concatenate_datasets
from lion_pytorch import Lion
from peft import (
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

true_tokens = "consistent"
false_tokens = "inconsistent"


def train(
    base_model: str = "", 
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    train_on_inputs: bool = True, 
    add_eos_token: bool = False,
    group_by_length: bool = False,
    label_smoothing_factor: float = None, 
    classification_alpha: float = 0.5,
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  
    wandb_log_model: str = "", 
    resume_from_checkpoint: str = None,  
    prompt_template_name: str = "alpaca",
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"label_smoothing_factor: {label_smoothing_factor}\n"
            f"classification_alpha: {classification_alpha}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    @dataclass
    class SmoothingTrainingArguments(transformers.TrainingArguments):
        classification_alpha: float = field(default=0.5)

    prompter = Prompter(prompt_template_name)

    device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    print(model)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0
    )
    tokenizer.padding_side = "right"

    true_id = tokenizer(true_tokens)["input_ids"][1]
    false_id = tokenizer(false_tokens)["input_ids"][1]
    print(f"true_id:{true_id},false_id:{false_id}")

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)

        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        tokenized_user_prompt = tokenize(
            user_prompt, add_eos_token=add_eos_token
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        if add_eos_token:
            user_prompt_len -= 1

        mask_input_ids = [
                              -100
                          ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                user_prompt_len:
                                                    ]
        if not train_on_inputs:
            tokenized_full_prompt["labels"] = mask_input_ids

        true_index = (mask_input_ids+[true_id]).index(true_id)
        false_index = (mask_input_ids+[false_id]).index(false_id)

        tokenized_full_prompt["labels"] = [min(true_index, false_index)] + tokenized_full_prompt["labels"]
        tokenized_full_prompt["input_ids"] = [min(true_index, false_index)] + tokenized_full_prompt["input_ids"]
        if true_index == false_index:
            tokenized_full_prompt["labels"] = []

        return tokenized_full_prompt

    model.enable_input_require_grads()
    model.save_checkpoint = model.save_pretrained

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )
            resume_from_checkpoint = (
                False
            )

        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle(2024).map(generate_and_tokenize_prompt)
        )
        train_data = train_data.filter(lambda example: len(example['labels']) > 1)
        val_data = (
            train_val["test"].shuffle(2024).map(generate_and_tokenize_prompt)
        )
        val_data = val_data.filter(lambda example: len(example['labels']) > 1)
    else:
        train_data = data["train"].shuffle(2024).map(generate_and_tokenize_prompt)
        train_data = train_data.filter(lambda example: len(example['labels']) > 1)
        val_data = None

    optimizer = Lion(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        use_triton=True
    )

    len_dataset = len(train_data)
    total_steps = (len_dataset // batch_size) * num_epochs if len_dataset % batch_size == 0 \
        else (len_dataset // batch_size + 1) * num_epochs

    schedule = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=total_steps
    )

    trainer = BalanceTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=SmoothingTrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=5,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=100 if val_set_size > 0 else None,
            save_steps=100,
            output_dir=output_dir,
            save_total_limit=32,
            load_best_model_at_end=True if val_set_size > 0 else False,
            group_by_length=group_by_length,
            label_smoothing_factor=label_smoothing_factor,
            classification_alpha=classification_alpha,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        optimizers=(optimizer, schedule),
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)