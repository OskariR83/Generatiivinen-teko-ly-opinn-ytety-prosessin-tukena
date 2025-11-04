"""
retrieval.py (strict v3.5)
--------------------------
Tiukka relevanssisuodatin. 
Palauttaa vain ne kappaleet, jotka ovat semanttisesti lähellä kysymystä.
Jos yhtäkään ei löydy → palautetaan tyhjä lista.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


RELEVANCE_THRESHOLD = 0.40  # liian matala = hylätään (arvo 0–1)

def retrieve_passages(query: str, index, passages: list[str], k: int = 5):
    print(f"🔎 Strict Retrieval v3.5 – kysymys: {query}\n")

    model_name = "TurkuNLP/sbert-cased-finnish-paraphrase"
    embedder = SentenceTransformer(model_name)

    q_emb = embedder.encode([query], normalize_embeddings=True)

    # Hae top-20 FAISS-tulos
    scores, idxs = index.search(np.array(q_emb, dtype=np.float32), 20)
    raw_candidates = [(scores[0][i], passages[idxs[0][i]]) for i in range(len(idxs[0]))]

    # Suodata pois epäolennaiset (matala semanttinen piste)
    filtered = [
        (score, text)
        for (score, text) in raw_candidates
        if score >= RELEVANCE_THRESHOLD
    ]

    if not filtered:
        print("⚠️ Ei yhtään riittävän relevanttia kappaletta. Palautetaan tyhjä lista.\n")
        return []

    # Lajittele pisteiden mukaan
    filtered.sort(key=lambda x: x[0], reverse=True)

    top_texts = [t for _, t in filtered][:k]
    print(f"✅ Löydetty {len(top_texts)} relevanttia kappaletta.\n")
    return top_texts
