"""
retrieval.py
-------------
Semanttinen haku TurkuNLP/sbert-base-finnish-paraphrase -mallilla.
Palauttaa parhaiten vastaavat tekstikappaleet FAISS-indeksistä.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


def expand_query(query: str) -> str:
    """Lisää synonyymivahvistusta hakulauseeseen, jos tunnistetaan tiettyjä avainsanoja."""
    q = query.lower()
    if "verkkolähde" in q or "lähde" in q or "lähdeluettelo" in q:
        query += " lähdeviite viittaaminen lähdeluettelo nettilähde internet-lähde viitattu lähdemerkintä"
    if "viite" in q:
        query += " lähdeviite kirjallisuusluettelo opinnäytetyö lähdeluettelo"
    return query


def retrieve_passages(query: str, index, passages: list[str], k: int = 5):
    """
    Hakee semanttisesti samankaltaiset kappaleet FAISS-indeksistä.
    Käyttää TurkuNLP/sbert-base-finnish-paraphrase -mallia kysymyksen embeddingin luomiseen.
    """
    print(f"🔎 Haetaan {k} parasta kappaletta kysymykseen: {query}")

    # 1️⃣ Laajenna hakulause synonyymeillä
    expanded_query = expand_query(query)

    # 2️⃣ Lataa suomalainen SBERT-malli
    model_name = "TurkuNLP/sbert-cased-finnish-paraphrase"
    embedder = SentenceTransformer(model_name)

    # 3️⃣ Luo embedding kysymyksestä ja tee haku
    q_emb = embedder.encode([expanded_query], normalize_embeddings=True)
    scores, idxs = index.search(np.array(q_emb, dtype=np.float32), k)

    # 4️⃣ Hae osuvat kappaleet
    retrieved = [passages[i] for i in idxs[0] if i < len(passages)]

    print(f"✅ {len(retrieved)} relevanttia kappaletta löydetty.\n")
    return retrieved
