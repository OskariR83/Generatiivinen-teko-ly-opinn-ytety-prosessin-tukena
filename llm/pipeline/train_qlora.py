import os
import json
from typing import Dict, Any

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


# ------------------------------------------------------
# Konfigin luku
# ------------------------------------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------
# Promptin muodostus datasetistä
#   - dataset-kentät: instruction, input, output
#   - luodaan yhtenäinen kysymys/konteksti/vastaus -formaatti
# ------------------------------------------------------
def make_prompt(example: Dict[str, str]) -> Dict[str, str]:
    instr = example["instruction"].strip()
    inp = example.get("input", "").strip()
    out = example["output"].strip()

    if inp:
        prompt = (
            "Alla on Savonian opinnäytetyöhön liittyvä kysymys ja lisäkonteksti.\n\n"
            f"KYSYMYS:\n{instr}\n\n"
            f"KONTEKSTI:\n{inp}\n\n"
            "VASTAUS:"
        )
    else:
        prompt = (
            "Alla on Savonian opinnäytetyöhön liittyvä kysymys.\n\n"
            f"KYSYMYS:\n{instr}\n\n"
            "VASTAUS:"
        )

    # Talteen sekä prompt että varsinainen vastausteksti
    example["prompt"] = prompt
    example["answer"] = out
    return example


# ------------------------------------------------------
# Tokenisointi + label-maskaus
#
#  - input_ids = [prompt-tokenit] + [answer-tokenit]
#  - labels    = [-100, ..., -100] + [answer-tokenit]
#
#  => malli näkee promptin, mutta loss lasketaan vain vastauksesta
# ------------------------------------------------------
def make_tokenize_fn(tokenizer, max_seq_length: int):
    def _tokenize(example: Dict[str, str]) -> Dict[str, Any]:
        prompt = example["prompt"]
        answer = example["answer"]

        # Tokenisoidaan erikseen, ilman extra special tokeneita
        prompt_tokens = tokenizer(
            prompt,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )
        answer_tokens = tokenizer(
            answer,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )

        input_ids = prompt_tokens["input_ids"] + answer_tokens["input_ids"]
        attention_mask = [1] * len(input_ids)

        # Truncataan jos menee yli
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            attention_mask = attention_mask[:max_seq_length]

        # Labels: prompt osa maskataan pois (-100), vastaus osa säilyy
        labels = (
            [-100] * len(prompt_tokens["input_ids"])
            + answer_tokens["input_ids"]
        )
        if len(labels) > max_seq_length:
            labels = labels[:max_seq_length]

        # Padding
        pad_len = max_seq_length - len(input_ids)
        if pad_len > 0:
            pad_id = tokenizer.pad_token_id
            input_ids += [pad_id] * pad_len
            attention_mask += [0] * pad_len
            labels += [-100] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return _tokenize


def main():
    # --------------------------------------------------
    # 1) Lue konfigi
    # --------------------------------------------------
    cfg = load_config("llm/pipeline/config_qlora_viking7b.json")

    base_model = cfg["base_model"]  # esim. "mpasila/Alpacazord-Viking-7B"
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

    # --------------------------------------------------
    # 2) Dataset
    # --------------------------------------------------
    print("🔹 Ladataan datasetit...")
    data_files = {"train": train_path, "validation": val_path}
    dataset = load_dataset("json", data_files=data_files)

    print("🔹 Muodostetaan promptit...")
    dataset = dataset.map(make_prompt)

    # --------------------------------------------------
    # 3) Tokenizer
    # --------------------------------------------------
    print("🔹 Ladataan tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------
    # 4) 4-bit QLoRA-perusmalli
    # --------------------------------------------------
    print("🔹 Ladataan 4-bit QLoRA-malli...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # A100: bf16 ok
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_cfg,
        device_map="auto",
    )

    # Valmistellaan k-bit-koulutusta varten
    model = prepare_model_for_kbit_training(model)

    # --------------------------------------------------
    # 5) LoRA-konfiguraatio
    #    (tyypilliset LLaMA/Viking-projekti-moduulit)
    # --------------------------------------------------
    lora_cfg = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # --------------------------------------------------
    # 6) Tokenisointi + label-maskaus
    # --------------------------------------------------
    print("🔹 Tokenisoidaan data...")
    tokenize_fn = make_tokenize_fn(tokenizer, max_seq_length)

    tokenized = dataset.map(
        tokenize_fn,
        batched=False,
        remove_columns=[
            "instruction",
            "input",
            "output",
            "prompt",
            "answer",
        ],
        desc="Tokenizing dataset",
    )

    # --------------------------------------------------
    # 7) TrainingArguments
    # --------------------------------------------------
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

        fp16=False,        # A100: käytetään bf16
        bf16=True,

        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",

        # Halutessasi:
        # gradient_checkpointing=True,
    )

    # --------------------------------------------------
    # 8) Trainer
    # --------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
    )

    # --------------------------------------------------
    # 9) Koulutus
    # --------------------------------------------------
    print("🚀 Aloitetaan QLoRA-koulutus...")
    trainer.train()

    # --------------------------------------------------
    # 10) Tallennus (LoRA-adapteri + tokenizer)
    # --------------------------------------------------
    print("💾 Tallennetaan LoRA-adapterit...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("✅ QLoRA-koulutus valmis!")


if __name__ == "__main__":
    main()
