"""
retrieval.py — Strict Retrieval
------------------------------------

Tämä moduuli vastaa seuraavista:

1) SBERT-embedding-mallin lataus (cache)
2) FAISS-pohjainen semanttinen haku
3) Tiukka relevanssisuodatus (vain riittävän lähellä oleva teksti kelpaa)
4) Duplikaattien ja matalan laadun poistaminen
5) Palauttaa vain korkeimman pistemäärän omaavat kappaleet

Huom: Jos yksikään kappale ei ylitä relevanssikynnystä,
      palautetaan tyhjä lista → Strict RAG hylkää kysymyksen.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


# ==================================================
# Globaalit asetukset
# ==================================================

RELEVANCE_THRESHOLD = 0.40  # min. semanttinen vastaavuus
_EMBEDDER = None



# ==================================================
# Embedding-kone (cache)
# ==================================================

def get_embedder():
    """Lataa embedding-mallin vain kerran."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("TurkuNLP/sbert-cased-finnish-paraphrase")
    return _EMBEDDER



# ==================================================
# Retrieval-toiminto
# ==================================================

def retrieve_passages(query: str, index, passages: list[str], k: int = 5):
    """
    Palauttaa korkeintaan k kappaletta, jotka:

    - FAISS haun mukaan ovat lähimpänä kysymystä
    - ja joiden semanttinen pistemäärä (SBERT) >= RELEVANCE_THRESHOLD

    Jos yhtään relevanttia ei löydy, palautetaan tyhjä lista.
    """

    print(f"🔎 Strict Retrieval v4.0 — Kysymys: {query}\n")

    embedder = get_embedder()

    # 1) Embed kysymys
    q_emb = embedder.encode([query], normalize_embeddings=True)

    # 2) FAISS-haku — haetaan väljemmin top-20
    scores, idxs = index.search(np.array(q_emb, dtype=np.float32), 20)

    # Järjestä FAISS-ehdokkaat (score + teksti)
    candidates = []
    for i in range(len(idxs[0])):
        score = float(scores[0][i])
        passage_idx = int(idxs[0][i])
        text = passages[passage_idx]
        candidates.append((score, text))

    # 3) Tiukka relevanssisuodatin
    filtered = [
        (score, text)
        for (score, text) in candidates
        if score >= RELEVANCE_THRESHOLD
    ]

    if not filtered:
        print("⚠️ Ei yhtään riittävän relevanttia kappaletta. Palautetaan tyhjä lista.\n")
        return []

    # 4) Lajittele korkeimmasta matalimpaan
    filtered.sort(key=lambda x: x[0], reverse=True)

    # 5) Poista duplikaatit (jos FAISS antaa samoja pätkiä eri kohdista)
    seen = set()
    unique_passages = []

    for _, text in filtered:
        key = text.strip()
        if key not in seen:
            seen.add(key)
            unique_passages.append(text)
        if len(unique_passages) >= k:
            break

    print(f"✅ Löydetty {len(unique_passages)} relevanttia kappaletta.\n")
    return unique_passages
