"""
docs_routes.py
---------------
FastAPI endpoints for document preprocessing and FAISS index status checking.

Includes:
1) POST /api/docs/preprocess  → Suorittaa OCR:n ja tekstiprosessoinnin kaikille dokumenteille
2) GET  /api/docs/status      → Tarkistaa prosessoitujen tiedostojen ja indeksin tilan
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
import sys

# Lisää LLM-moduulin polku (projektin juureen asti)
BASE_PATH = Path(__file__).resolve().parents[3]  # nousee backend/app/api -> project/
LLM_SRC = BASE_PATH / "llm" / "src"
if str(LLM_SRC) not in sys.path:
    sys.path.insert(0, str(LLM_SRC))

# LLM-moduulit
from ocr_utils import preprocess_all_documents
from indexing import build_faiss_index

router = APIRouter(prefix="/docs", tags=["Documents"])


@router.post("/preprocess")
async def preprocess_documents():
    """
    Suorittaa kaikkien dokumenttien OCR:n ja tekstiprosessoinnin.
    Käyttää Unstructured.io + PaddleOCR pipelinea.
    Rakentaa myös FAISS-indeksin valmiiksi.
    """
    try:
        print("🧠 Starting document preprocessing...\n")
        preprocess_all_documents()
        print("\n✅ Document preprocessing completed.\n")

        print("🏗️ Building FAISS index...\n")
        build_faiss_index()

        return {"status": "success", "message": "Documents processed and FAISS index updated."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def check_status():
    """
    Tarkistaa onko prosessoituja tiedostoja ja FAISS-indeksi olemassa.
    Palauttaa tiedon JSON-muodossa.
    """
    try:
        docs_dir = BASE_PATH / "docs"
        processed_dir = docs_dir / "processed"
        index_dir = docs_dir / "indexes"
        log_file = docs_dir.parent / "logs" / "ocr_failures.log"

        processed_files = list(processed_dir.glob("*.json"))
        index_files = list(index_dir.glob("*.faiss"))
        log_exists = log_file.exists()
        log_content = ""

        if log_exists:
            with open(log_file, "r", encoding="utf-8") as f:
                log_content = f.read()[-2000:]  # näytetään viimeiset ~2000 merkkiä

        return {
            "processed_count": len(processed_files),
            "index_exists": len(index_files) > 0,
            "processed_dir": str(processed_dir),
            "index_dir": str(index_dir),
            "ocr_log_present": log_exists,
            "ocr_log_tail": log_content,
            "status": "ok"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
