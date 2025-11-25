"""
generation.py — STRICT RAG v4.5 + QLoRA Adapter + Global Cache
--------------------------------------------------------------
LLM saa käyttää vain dokumenteista löytyviä lauseita.
Kaikki generoidut lauseet validoidaan embedding-mallilla.
QLoRA-adapteri ladataan mukaan Viking-7B -malliin.
Malli ladataan vain kerran (cache), ei jokaisella kysymyksellä.
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import os
import re


# ------------------------------------
# Thresholdit
# ------------------------------------
SEMANTIC_MATCH_THRESHOLD = 0.40
SENTENCE_VALIDATE_THRESHOLD = 0.45


# ------------------------------------
# Embedding-malli (cached)
# ------------------------------------
_EMBEDDER = None

def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("TurkuNLP/sbert-cased-finnish-paraphrase")
    return _EMBEDDER


# ------------------------------------
# LLM + LoRA-adapterin global cache
# ------------------------------------
_llm_model = None
_llm_tokenizer = None

def get_llm():
    global _llm_model, _llm_tokenizer

    if _llm_model is not None:
        return _llm_model, _llm_tokenizer

    base_model_name = "mpasila/Alpacazord-Viking-7B"

    # käytä ABSOLUUTTISTA POLKUA
    adapter_path = "/home/user/GENERATIIVINEN-TEKOALY-OPINNAYTETYOPROSESSIN-TUKENA/llm/pipeline/output/viking7b-qlora-ont"

    print("Ladataan perusmalli...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("Ladataan LoRA-adapteri...")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_local=True,   # pakota paikallinen haku
        torch_dtype=torch.float16,
    )

    model.eval()

    _llm_model = model
    _llm_tokenizer = tokenizer
    return _llm_model, _llm_tokenizer

# ------------------------------------
# Semanttinen osuvuus
# ------------------------------------
def context_is_relevant(question: str, passages: list[str]) -> bool:
    if not passages:
        return False

    emb = get_embedder()
    q_vec = emb.encode([question], normalize_embeddings=True)[0]

    for p in passages:
        p_vec = emb.encode([p], normalize_embeddings=True)[0]
        score = float(np.dot(q_vec, p_vec))

        if score >= SEMANTIC_MATCH_THRESHOLD:
            return True

    return False


# ------------------------------------
# Pilko lauseisiin
# ------------------------------------
def split_sentences(text: str) -> list[str]:
    s = re.split(r'(?<=[.!?])\s+', text.strip())
    return [x.strip() for x in s if x.strip()]


# ------------------------------------
# Vastauksen lausevalidointi
# ------------------------------------
def validate_answer_sentences(answer: str, context: list[str]) -> bool:
    emb = get_embedder()

    context_vecs = emb.encode(context, normalize_embeddings=True)
    sentences = split_sentences(answer)

    print(f"Validointiin meneviä lauseita: {len(sentences)}")

    for s in sentences:
        s_vec = emb.encode([s], normalize_embeddings=True)[0]
        sims = np.dot(context_vecs, s_vec)
        max_sim = float(np.max(sims))

        print(f"  • Lause → max similarity = {max_sim:.3f}")

        if max_sim < SENTENCE_VALIDATE_THRESHOLD:
            print(f"❌ Hylätty lause: {s}")
            return False

    return True


# ------------------------------------
# Generointi QLoRA-adapterilla
# ------------------------------------
def generate_answer(question: str, context: list[str]) -> str:

    # 1) Ei kontekstia → ei vastausta
    if not context:
        return "En löydä varmaa ohjetta annetuista lähteistä."

    # 2) Tarkista semanttinen relevanssi
    if not context_is_relevant(question, context):
        return "En löydä varmaa ohjetta annetuista lähteistä."

    print("\nGeneroidaan vastaus (Strict RAG v4.5 + QLoRA)...")

    # 3) Koosta dokumenttikonteksti LLM:lle
    combined_context = "\n".join(context)

    # 4) Prompt
    prompt = (
        "Sinä olet opinnäytetyöavustaja. Vastaa kysymykseen käyttäen VAIN seuraavasta dokumentista "
        "löytyviä tietoja. Et saa keksiä mitään uutta.\n\n"
        "DOKUMENTTI:\n"
        f"{combined_context}\n\n"
        f"KYSYMYS: {question}\n\n"
        "VASTAUS:"
    )

    # 5) Lataa LLM ja tokenizer (cached)
    model, tokenizer = get_llm()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1500
    ).to(model.device)

    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")

    # 6) Generointi
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    answer = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    print(f"\n LLM vastaus (ensimmäiset 150 merkkiä): {answer[:150]}...")

    # 7) Lausevalidointi
    if not validate_answer_sentences(answer, context):
        return "En löydä varmaa ohjetta annetuista lähteistä."

    print("✅ Kaikki lauseet hyväksytty.")
    return answer
