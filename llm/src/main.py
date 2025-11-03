"""
main.py
--------
Käynnistää koko RAG-putken:
1️⃣ Prosessoi alkuperäiset dokumentit (jos tarpeen)
2️⃣ Rakentaa tai lataa FAISS-indeksin
3️⃣ Suorittaa haun ja generoi vastauksen
"""
import os
os.environ["ORT_DISABLE_TENSORRT"] = "1"
os.environ["ORT_TENSORRT_UNAVAILABLE_WARNINGS"] = "1"
os.environ["ORT_PROVIDERS"] = "CUDAExecutionProvider"
os.environ["LD_LIBRARY_PATH"] = ""


import sys
from pathlib import Path

# Lisää projektin juurihakemisto sys.pathiin (kaksi tasoa ylöspäin)
BASE_PATH = Path(__file__).resolve().parents[2]
if str(BASE_PATH) not in sys.path:
    sys.path.insert(0, str(BASE_PATH))

# Projektimoduulit
from llm.src.indexing import build_faiss_index
from llm.src.retrieval import retrieve_passages
from llm.src.generation import generate_answer
from llm.src.utils import clear_memory
from llm.src.ocr_utils import preprocess_all_documents


def main(question_override=None):
    print("🚀 Käynnistetään RAG-putki...\n")

    # 1️⃣ Tarkista ja suorita dokumenttien prosessointi (vain jos processed-kansio on tyhjä)
    processed_dir = BASE_PATH / "docs" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(processed_dir.glob("*.json"))
    if not json_files:
        print("🔍 Prosessoituja tiedostoja ei löytynyt — aloitetaan OCR-prosessointi...\n")
        preprocess_all_documents()
    else:
        print(f"✅ Löydetty {len(json_files)} valmiiksi prosessoitua tiedostoa. Ohitetaan OCR.\n")

    # 2️⃣ Rakenna tai lataa FAISS-indeksi
    result = build_faiss_index()
    if result is None:
        print("❌ FAISS-indeksin rakentaminen epäonnistui – varmista, että prosessointi onnistui.")
        return
    index, passages, metadata = result

    # 3️⃣ Hae vastaus
    question = question_override or "Miten verkkolähde merkitään lähdeluetteloon suomalaisessa opinnäytetyössä?"
    print(f"🔎 Haetaan 5 parasta kappaletta kysymykseen: {question}\n")

    top_passages = retrieve_passages(question, index, passages, k=5)
    if not top_passages:
        print("⚠️ Ei kappaleita analysoitavaksi.")
        return

    print("📄 Käytetyt kappaleet vastausta varten:\n")
    for i, kpl in enumerate(top_passages, start=1):
        print(f"[{i}] {kpl[:300]}...\n")

    # 4️⃣ Generoi vastaus
    answer = generate_answer(question, top_passages)

    print("\n" + "=" * 50)
    print("🎯 LOPULLINEN VASTAUS (Viking-13B)")
    print("=" * 50)
    print(f"\nKysymys: {question}")
    print(f"\nVastaus:\n{answer}")
    print("\n" + "=" * 50)


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            question = " ".join(sys.argv[1:])
            main(question)
        else:
            main()
    finally:
        clear_memory()
