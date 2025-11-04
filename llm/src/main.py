"""
main.py
--------
Käynnistää koko RAG-putken:

1) Dokumenttien OCR- ja tekstin esiprosessointi (jos ei cachea)
2) FAISS-indeksin rakentaminen tai lataaminen
3) Retrieval v3 – semanttinen + avainsanapainotteinen haku
4) Viking-7B (tai Viking-13B) vastaus generointi
5) Varautuminen tapaukselle, jossa viiteohjeita ei löydy

Tämä versio toimii yhdessä:
- retrieval.py (v3)
- generation.py (v2)
"""

import os
os.environ["ORT_DISABLE_TENSORRT"] = "1"
os.environ["ORT_TENSORRT_UNAVAILABLE_WARNINGS"] = "1"
os.environ["ORT_PROVIDER"] = "CUDAExecutionProvider"

import sys
from pathlib import Path

# Lisää projektin juurihakemisto pythonpathiin
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

    # ----------------------------
    # 1) Dokumenttien prosessointi
    # ----------------------------
    processed_dir = BASE_PATH / "docs" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(processed_dir.glob("*.json"))
    if not json_files:
        print("🔍 Prosessoituja tiedostoja ei löytynyt — aloitetaan OCR-prosessointi...\n")
        preprocess_all_documents()
    else:
        print(f"✅ Löydetty {len(json_files)} valmiiksi prosessoitua tiedostoa. Ohitetaan OCR.\n")

    # ----------------------------
    # 2) Rakenna tai lataa FAISS-index
    # ----------------------------
    result = build_faiss_index()
    if result is None:
        print("❌ FAISS-indeksin rakentaminen epäonnistui – varmista, että prosessointi onnistui.")
        return
    index, passages, metadata = result

    # ----------------------------
    # 3) Kysymys
    # ----------------------------
    question = question_override or "Miten valitsen sopivan tutkimusmenetelmän?"

    print(f"🔎 Haku: {question}\n")

    # ----------------------------
    # 4) Retrieval v3
    # ----------------------------
    top_passages = retrieve_passages(question, index, passages, k=5)
    if not top_passages:
        print("⚠️ Ei kappaleita analysoitavaksi.")
        return

    print("📄 Käytetyt kappaleet vastausta varten:\n")
    for i, kpl in enumerate(top_passages, start=1):
        print(f"[{i}] {kpl[:300]}...\n")

    # ----------------------------
    # 6) Generointi Viking-7B / Viking-13B
    # ----------------------------
    answer = generate_answer(question, top_passages)

    print("\n" + "=" * 50)
    print("🎯 LOPULLINEN VASTAUS")
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
