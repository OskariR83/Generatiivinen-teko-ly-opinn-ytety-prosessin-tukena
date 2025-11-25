"""
indexing.py
------------
Rakentaa FAISS-indeksin valmiiksi prosessoiduista teksteistä,
käyttäen TurkuNLP/sbert-cased-finnish-paraphrase -mallia.

🔹 Lukee JSON-tiedostoja muodossa: {"text": "..."}
🔹 Pilkkoo tekstin 150–300 sanan kappaleisiin
🔹 Luo FAISS-indeksin, joka tallennetaan tiedostoon
🔹 Palauttaa: (index, passages, metadata)
"""

import json
import numpy as np
import torch
import faiss
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, logging
from tqdm import tqdm

logging.set_verbosity_error()


def build_faiss_index(processed_dir=None, index_path=None):
    """
    Rakentaa FAISS-indeksin valmiiksi prosessoiduista teksteistä (docs/processed).
    Palauttaa (index, passages, metadata).
    """

    print("Rakennetaan FAISS-indeksi TurkuNLP/sbert-cased-finnish-paraphrase -mallilla...\n")

    base_dir = Path(__file__).resolve().parents[2]
    if processed_dir is None:
        processed_dir = base_dir / "docs/processed"
    if index_path is None:
        index_path = base_dir / "docs/indexes/combined_index.faiss"

    processed_dir = Path(processed_dir)
    index_file = Path(index_path)
    meta_file = index_file.with_suffix(".meta.json")
    index_file.parent.mkdir(parents=True, exist_ok=True)

    # 🔹 Ladataan malli
    model_name = "TurkuNLP/sbert-cased-finnish-paraphrase"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Käytössä laite: {device}\n")

    # 🔍 Luetaan kaikki JSON-tiedostot
    json_files = list(processed_dir.rglob("*.json"))
    if not json_files:
        print(f"⚠️ Ei löydetty JSON-tiedostoja hakemistosta: {processed_dir.resolve()}")
        print("❌ FAISS-indeksin rakentaminen epäonnistui – varmista, että OCR-prosessointi on tehty.")
        return None

    all_passages, metadata = [], []
    print(f"Ladataan {len(json_files)} prosessoitua tiedostoa...\n")

    for file in tqdm(json_files, desc="Luetaan aineistoa", unit="tiedosto", leave=False, dynamic_ncols=True):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "text" in data and isinstance(data["text"], str):
                    text = data["text"].strip()
                    if text:
                        words = text.split()
                        chunk_size = 200  #hyvä kompromissi semanttisen yhtenäisyyden ja tarkkuuden välillä
                        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
                        all_passages.extend(chunks)
                        metadata.extend([f"{file.name}#{i}" for i in range(len(chunks))])
        except Exception as e:
            print(f"⚠️ Virhe tiedostossa {file.name}: {e}")

    if not all_passages:
        print(f"❌ Ei kelvollista tekstiä luettavissa hakemistosta {processed_dir}")
        return None

    print(f"\n✅ Lataus valmis. Tekstikappaleita yhteensä: {len(all_passages)}.\n")

    #Lasketaan embeddingit
    embeddings = []
    print("Lasketaan embeddingit (vektoriesitykset)...\n")

    batch_size = 8
    for i in tqdm(range(0, len(all_passages), batch_size),
                  desc="Embedding-erät", unit="batch", leave=False, dynamic_ncols=True):

        batch = all_passages[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            vecs = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        embeddings.extend(vecs)

        # Print progress occasionally
        if (i // batch_size) % 100 == 0 and i > 0:
            print(f"   → {i}/{len(all_passages)} kappaletta käsitelty...")

    embeddings = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)

    #Luo FAISS-indeksi
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    #Tallennus
    faiss.write_index(index, str(index_file))
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata}, f, ensure_ascii=False, indent=2)

    print(f"\n💾 FAISS-indeksi tallennettu: {index_file}")
    print(f"💾 Metatiedot tallennettu: {meta_file}")
    print("Indeksin rakennus valmis.\n")

    return index, all_passages, metadata


if __name__ == "__main__":
    build_faiss_index()
