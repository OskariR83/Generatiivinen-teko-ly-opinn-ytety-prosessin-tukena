"""
generation.py — STRICT RAG v4.5 (Hybrid Compose)
-----------------------------------------------
LLM saa käyttää VAIN dokumentista hyväksyttyjä lauseita.
Jokainen lause validoidaan embedding-mallilla.
Jos yksikin generoidun vastauksen lause EI läpäise validointia,
AIKAISIN palautetaan: "En löydä varmaa ohjetta annetuista lähteistä."
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer


# ------------------------------------
# ⚙️ Thresholdit
# ------------------------------------
SEMANTIC_MATCH_THRESHOLD = 0.40   # minimi että retrieval on osuva
SENTENCE_VALIDATE_THRESHOLD = 0.45  # jokainen vastauslause validoidaan dokumenttia vasten


# ------------------------------------
# 🧠 Embedding-malli (cache)
# ------------------------------------
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("TurkuNLP/sbert-cased-finnish-paraphrase")
    return _embedder


# ------------------------------------
# 🔎 Semanttinen osuvuus (koko kontekstille)
# ------------------------------------
def context_is_relevant(question: str, passages: list[str]) -> bool:
    if not passages:
        return False

    embedder = get_embedder()

    q_vec = embedder.encode([question], normalize_embeddings=True)[0]

    for p in passages:
        p_vec = embedder.encode([p], normalize_embeddings=True)[0]
        score = float(np.dot(q_vec, p_vec))

        if score >= SEMANTIC_MATCH_THRESHOLD:
            return True

    return False


# ------------------------------------
# ✂️ Pilko lauseisiin
# ------------------------------------
def split_sentences(text: str) -> list[str]:
    import re
    s = re.split(r'(?<=[.!?])\s+', text.strip())
    return [x.strip() for x in s if x.strip()]


# ------------------------------------
# 🧪 Lausekohtainen validointi
# ------------------------------------
def validate_answer_sentences(answer: str, context: list[str]) -> bool:
    embedder = get_embedder()

    context_vecs = embedder.encode(context, normalize_embeddings=True)

    sentences = split_sentences(answer)
    print(f"🔎 Validointiin meneviä lauseita: {len(sentences)}")

    for s in sentences:
        s_vec = embedder.encode([s], normalize_embeddings=True)[0]

        sims = np.dot(context_vecs, s_vec)
        max_sim = float(np.max(sims))

        print(f"  • Lause → max similarity = {max_sim:.3f}")

        if max_sim < SENTENCE_VALIDATE_THRESHOLD:
            print(f"❌ Hylätty lause: {s}")
            return False

    return True


# ------------------------------------
# 🤖 Generointi (Hybrid Compose)
# ------------------------------------
def generate_answer(question: str, context: list[str]) -> str:

    # 1) Ei kontekstia
    if not context:
        return "En löydä varmaa ohjetta annetuista lähteistä."

    # 2) Tarkista että kysymys liittyy edes yhteen kappaleeseen
    if not context_is_relevant(question, context):
        return "En löydä varmaa ohjetta annetuista lähteistä."

    print("\n⚙️ Generoidaan vastaus mallilla Viking-7B (deterministinen, strict v4.5)...")

    # 3) Koosta dokumenttikonteksti LLM:lle
    source_context = "\n".join(context)

    # 4) Prompt
    prompt = (
        "Sinä olet opinnäytetyöavustaja. Vastaa kysymykseen käyttäen VAIN seuraavasta dokumentista "
        "löytyviä tietoja. Et saa keksiä mitään uutta: kaikki väitteet tulee löytyä dokumentista.\n\n"
        "DOKUMENTTI:\n"
        f"{source_context}\n\n"
        f"KYSYMYS: {question}\n\n"
        "VASTAUS (tiiviisti ja hyvällä suomen kielellä):"
    )

    # 5) LLM
    model_name = "mpasila/Alpacazord-Viking-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1500
    ).to(model.device)

    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,      # deterministinen
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 6) Decode
    answer = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    print(f"\n📝 LLM vastaus ({len(answer)} merkkiä): {answer[:120]}...")

    # 7) Turvatarkastus #2: lausevalidointi
    if not validate_answer_sentences(answer, context):
        return "En löydä varmaa ohjetta annetuista lähteistä."

    print("✅ Kaikki lauseet validoitu — vastaus hyväksytty.")
    return answer
