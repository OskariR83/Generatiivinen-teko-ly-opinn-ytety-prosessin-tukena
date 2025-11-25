import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(question: str, context: str = ""):
    """
    Muodostaa TÄSMÄLLEEN saman promptin,
    jota train_qlora.py käyttää.
    """

    if context.strip():
        return (
            "Alla on ohje ja siihen liittyvä lisäkonteksti.\n\n"
            f"Ohje:\n{question}\n\n"
            f"Konteksti:\n{context}\n\n"
            "Kirjoita selkeä, lyhyt ja Savonian ohjeisiin nojaava vastaus.\n\nVastaus:\n"
        )
    else:
        return (
            "Alla on opiskelijan kysymys liittyen Savonian opinnäytetyöohjeisiin.\n\n"
            f"Kysymys:\n{question}\n\n"
            "Kirjoita selkeä, lyhyt ja Savonian ohjeisiin nojaava vastaus.\n\nVastaus:\n"
        )


def load_model_and_tokenizer(base_model: str, lora_path: str):
    print("🔹 Ladataan tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("🔹 Ladataan kvantisoitu 4-bit perusmalli...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_cfg,
        device_map="auto",
    )

    print(f"🔹 Ladataan LoRA-adapteri: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    return model, tokenizer


def generate(model, tokenizer, question: str, context: str = "", max_new_tokens=300):
    prompt = build_prompt(question, context)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Palauta vain vastaus–osa
    if "Vastaus:" in decoded:
        return decoded.split("Vastaus:")[1].strip()
    return decoded


def main():
    cfg = load_config("llm/pipeline/config_qlora_viking7b.json")

    base_model = cfg["base_model"]
    lora_path = cfg["output_dir"]

    print("🚀 Ladataan QLoRA-malli inferenceä varten...")
    model, tokenizer = load_model_and_tokenizer(base_model, lora_path)

    print("\n❓ Kysy mitä vain Savonian opinnäytetyöohjeisiin liittyen (tyhjä = lopetus)")
    while True:
        q = input("\nKysymys: ").strip()
        if not q:
            break

        answer = generate(model, tokenizer, q)
        print("\n💬 Mallin vastaus:\n")
        print(answer)
        print("\n" + "-" * 60)


if __name__ == "__main__":
    main()
