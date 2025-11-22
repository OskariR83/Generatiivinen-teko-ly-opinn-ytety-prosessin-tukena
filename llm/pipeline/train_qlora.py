import os
import json

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_prompt(example):
    """
    Muodostaa "Alpaca-tyyppisen" promptin yhdestä rivistä.
    """
    instr = example["instruction"].strip()
    inp = example.get("input", "").strip()
    out = example["output"].strip()

    if inp:
        prompt = (
            "Alla on ohje ja siihen liittyvä lisäkonteksti.\n\n"
            f"Ohje:\n{instr}\n\n"
            f"Konteksti:\n{inp}\n\n"
            "Kirjoita selkeä, lyhyt ja Savonian ohjeisiin nojaava vastaus.\n\nVastaus:\n"
        )
    else:
        prompt = (
            "Alla on opiskelijan kysymys liittyen Savonian opinnäytetyöohjeisiin.\n\n"
            f"Kysymys:\n{instr}\n\n"
            "Kirjoita selkeä, lyhyt ja Savonian ohjeisiin nojaava vastaus.\n\nVastaus:\n"
        )

    example["prompt"] = prompt
    example["labels"] = out
    return example


def tokenize(tokenizer, max_seq_length):
    def _tokenize(example):
        text = example["prompt"] + example["labels"]
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    return _tokenize


def main():
    cfg = load_config("llm/pipeline/config_qlora_viking7b.json")

    base_model = cfg["base_model"]
    train_path = cfg["train_path"]
    val_path = cfg["val_path"]
    output_dir = cfg["output_dir"]

    max_seq_length = cfg.get("max_seq_length", 2048)
    num_train_epochs = cfg.get("num_train_epochs", 3)
    per_device_train_batch_size = cfg.get("per_device_train_batch_size", 1)
    gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 16)
    learning_rate = cfg.get("learning_rate", 2e-4)
    warmup_ratio = cfg.get("warmup_ratio", 0.03)
    weight_decay = cfg.get("weight_decay", 0.01)
    logging_steps = cfg.get("logging_steps", 10)
    save_steps = cfg.get("save_steps", 200)
    eval_steps = cfg.get("eval_steps", 200)

    os.makedirs(output_dir, exist_ok=True)

    print("🔹 Ladataan datasetit...")
    data_files = {"train": train_path, "validation": val_path}
    dataset = load_dataset("json", data_files=data_files)

    dataset = dataset.map(make_prompt)

    print("🔹 Ladataan tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("🔹 Kvantisoidaan malli 4-bit (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("🔹 Tokenisoidaan data...")
    tokenized = dataset.map(
        tokenize(tokenizer, max_seq_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
    )

    print("🚀 Aloitetaan QLoRA-koulutus...")
    trainer.train()

    print("💾 Tallennetaan LoRA-adapterit...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("✅ Valmis!")


if __name__ == "__main__":
    main()
