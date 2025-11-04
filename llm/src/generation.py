"""
generation.py (strict v3.1)
---------------------------
Tiukin mahdollinen malli: jos konteksti ei ole selvästi aiheeseen liittyvä,
LLM:ää EI kutsuta lainkaan.
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer


SEMANTIC_MATCH_THRESHOLD = 0.40  # 0–1


def _semantic_match(question: str, passages: list[str]) -> bool:
    """True jos yksikin kappale on semanttisesti lähellä kysymystä."""
    if not passages:
        return False

    embedder = SentenceTransformer("TurkuNLP/sbert-cased-finnish-paraphrase")

    q_vec = embedder.encode([question], normalize_embeddings=True)[0]

    for p in passages:
        p_vec = embedder.encode([p], normalize_embeddings=True)[0]
        score = float(np.dot(q_vec, p_vec))

        if score >= SEMANTIC_MATCH_THRESHOLD:
            return True

    return False


def generate_answer(question: str, context: list[str]) -> str:

    # 1) Tyhjä konteksti → ei vastausta
    if not context:
        return "En löydä varmaa ohjetta annetuista lähteistä."

    # 2) Semanttinen match check
    if not _semantic_match(question, context):
        return "En löydä varmaa ohjetta annetuista lähteistä."

    print("\n⚙️ Generoidaan vastaus mallilla Viking-7B...")

    model_name = "mpasila/Alpacazord-Viking-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()

    # 3) Koosta lähdeteksti
    source_text = "\n\n".join(context)

    # 4) Tiukka system prompt
    prompt = (
        "Vastaa seuraavaan kysymykseen käyttäen VAIN annettua lähdeaineistoa.\n"
        "Jos vastausta ei löydy lähdeaineistosta: sano täsmälleen:\n"
        "'En löydä varmaa ohjetta annetuista lähteistä.'\n\n"
        f"Kysymys: {question}\n\n"
        f"Lähdeaineisto:\n{source_text}\n\n"
        "Vastaus:"
    )

    # 5) Tokenointi
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 6) Generointi (turvallinen, minimaalinen)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=180,
            temperature=0.25,
            top_p=0.85,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer = tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()

    # 7) Jos vastaus on liian lyhyt → fallback
    if len(answer) < 10:
        return "En löydä varmaa ohjetta annetuista lähteistä."

    return answer
