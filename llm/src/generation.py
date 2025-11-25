"""
generation.py — Strict RAG + QLoRA-adapterille optimoitu generointi
------------------------------------------------------------------------

Tämä moduuli vastaa seuraavista toiminnoista:

1) LoRA-adapterilla laajennetun Viking-7B -mallin lataus (cache, ladataan vain kerran)
2) RAG-vastausten generointi deterministisesti (ei hallusinaatiota)
3) Sekä kysymystä että dokumenttikontekstia hyödyntävä validointi
4) Lause- ja sanakohtainen tulosten siivous
5) Lopullisen vastauksen rajaaminen 3–5 virkkeeseen

Generointi noudattaa Strict RAG -periaatteita:
- Ei saa keksiä uutta tietoa
- Vastausten täytyy nojata annetun dokumentin sisältöön
- Vain kysymykseen liittyvät lauseet hyväksytään
"""

import os
import re
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from peft import PeftModel


# ==================================================
# Globaalit välimuistit
# ==================================================

_EMBEDDER = None
_LLM = None
_TOKENIZER = None


# ==================================================
# Konfiguraatiot
# ==================================================

SEMANTIC_MATCH_THRESHOLD = 0.40
SENTENCE_VALIDATE_THRESHOLD = 0.45



# ==================================================
# Embedding-mallin lataus
# ==================================================

def get_embedder():
    """Lataa SBERT-mallin vain kerran (cache)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("TurkuNLP/sbert-cased-finnish-paraphrase")
    return _EMBEDDER



# ==================================================
# LLM + LoRA-adapterin lataus
# ==================================================

def get_llm():
    """
    Lataa Viking-7B + QLoRA-adapteri vain ensimmäisellä kutsulla.
    Kaikki myöhemmät kutsut käyttävät cachea.
    """
    global _LLM, _TOKENIZER

    if _LLM is not None:
        return _LLM, _TOKENIZER

    base_model = "mpasila/Alpacazord-Viking-7B"
    adapter_path = (
        "/home/user/GENERATIIVINEN-TEKOALY-OPINNAYTETYOPROSESSIN-TUKENA/"
        "llm/pipeline/output/viking7b-qlora-ont"
    )

    print("Ladataan Viking-7B -perusmalli...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Truncation-korjaus (estää rikkinäiset aloitussanat)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("Ladataan LoRA-adapteri...")
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        torch_dtype=torch.float16,
        is_local=True
    )

    model.eval()

    _LLM = model
    _TOKENIZER = tokenizer

    return _LLM, _TOKENIZER



# ==================================================
# Apufunktiot (sentence split, cleanup jne.)
# ==================================================

def split_sentences(text: str):
    """Jakaa tekstin virkkeisiin, huomioi ., !, ?"""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def cleanup_truncated_word(answer: str) -> str:
    """Poistaa viimeisen sanan, jos se näyttää katkenneelta."""
    if not answer.endswith(('.', '!', '?')):
        parts = answer.split()
        if parts:
            parts = parts[:-1]
        return " ".join(parts).strip()
    return answer


def limit_sentences(answer: str, max_sentences: int = 4) -> str:
    """Rajoittaa vastauksen enintään X virkkeeseen."""
    sents = split_sentences(answer)
    return " ".join(sents[:max_sentences]).strip()



# ==================================================
# Validointi: dokumentti + kysymys yhtä aikaa
# ==================================================

def validate_answer_sentences(answer: str, context: list[str], question: str) -> bool:
    """
    Hyväksyy lauseen vain jos:
      1) se on riittävän lähellä dokumenttikontekstia
      2) se on riittävän lähellä kysymystä
    """
    emb = get_embedder()

    context_vecs = emb.encode(context, normalize_embeddings=True)
    q_vec = emb.encode([question], normalize_embeddings=True)[0]

    sentences = split_sentences(answer)

    print(f"Validointiin meneviä lauseita: {len(sentences)}")

    for s in sentences:
        s_vec = emb.encode([s], normalize_embeddings=True)[0]

        sim_ctx = float(np.max(np.dot(context_vecs, s_vec)))
        sim_q = float(np.dot(q_vec, s_vec))

        print(f"• {s}")
        print(f"    - konteksti-sim = {sim_ctx:.3f}")
        print(f"    - kysymys-sim = {sim_q:.3f}")

        if sim_ctx < SENTENCE_VALIDATE_THRESHOLD or sim_q < 0.30:
            print(f"❌ Hylätty lause: {s}")
            return False

    return True



# ==================================================
# Päätoiminto: LLM-vastauksen generointi
# ==================================================

def generate_answer(question: str, context: list[str]) -> str:
    """Generoi vastauksen Viking-7B + QLoRA -mallilla Strict RAG -periaatteella."""

    # 1) Ei kontekstia → ei vastausta
    if not context:
        return "En löydä varmaa ohjetta annetuista lähteistä."

    # 2) Semanttinen tarkistus (estää täysin väärät haut)
    emb = get_embedder()
    q_vec = emb.encode([question], normalize_embeddings=True)[0]

    relevant = False
    for p in context:
        p_vec = emb.encode([p], normalize_embeddings=True)[0]
        if float(np.dot(q_vec, p_vec)) >= SEMANTIC_MATCH_THRESHOLD:
            relevant = True
            break

    if not relevant:
        return "En löydä varmaa ohjetta annetuista lähteistä."

    print("\nGeneroidaan vastaus Strict RAG...\n")

    # 3) Yhdistä hakukappaleet
    combined = "\n".join(context)

    # 4) Prompt (tiivis ja ohjaava)
    prompt = (
        "Sinä olet opinnäytetyöavustaja. Vastaa kysymykseen käyttämällä VAIN alla olevan "
        "dokumentin sisältöä. Älä keksi mitään uutta.\n\n"
        "DOKUMENTTI:\n"
        f"{combined}\n\n"
        f"KYSYMYS: {question}\n\n"
        "VASTAUS (3–5 virkettä, selkeä ja kokonainen vastaus):"
    )

    # 5) LLM + tokenizer
    model, tokenizer = get_llm()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=3000
    ).to(model.device)

    # Poistetaan token_type_ids jos malli ei tue niitä
    inputs.pop("token_type_ids", None)

    # 6) Generointi
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=False
        )

    answer = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    # 7) Cleanup
    answer = cleanup_truncated_word(answer)
    answer = limit_sentences(answer, max_sentences=6)

    print(f"\n LLM vastaus (debug): {answer[:200]}...\n")

    # 8) Validointi
    if not validate_answer_sentences(answer, context, question):
        return "En löydä varmaa ohjetta annetuista lähteistä."

    return answer
