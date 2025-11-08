"""
llm_routes.py
-------------
FastAPI endpoint for LLM-based RAG pipeline.
- Luo kontekstin kysymykselle (retrieval)
- Kutsuu Viking-7B -mallia vain jos löytyy relevanttia kontekstia
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sys
from pathlib import Path

# -------------------------------------------------------------
# 🧩 Lisää LLM-moduulin polku (backend/app/api → projektin juuri)
# -------------------------------------------------------------
BASE_PATH = Path(__file__).resolve().parents[3]
LLM_SRC = BASE_PATH / "llm" / "src"
if str(LLM_SRC) not in sys.path:
    sys.path.insert(0, str(LLM_SRC))

# -------------------------------------------------------------
# 🧠 LLM pipeline -komponentit
# -------------------------------------------------------------
from retrieval import retrieve_passages
from indexing import build_faiss_index
from generation import generate_answer
from utils import clear_memory

router = APIRouter(prefix="/llm", tags=["LLM"])


# -------------------------------------------------------------
# 📘 Pydantic-malli
# -------------------------------------------------------------
class LLMQuery(BaseModel):
    question: str


# -------------------------------------------------------------
# 💬 /api/llm/query endpoint
# -------------------------------------------------------------
@router.post("/query")
async def query_llm(data: LLMQuery):
    """
    Suorittaa RAG-pipelinekyselyn:
    1) Rakentaa tai hakee FAISS-indeksin
    2) Hakee relevantit tekstikappaleet
    3) Generoi tiiviin vastauksen Viking-7B -mallilla
    4) Palauttaa vastauksen JSON-muodossa
    """
    question = data.question.strip()
    print(f"\n🔎 New query received: {question}\n")

    try:
        # 1️⃣ Hae tai rakenna FAISS-indeksi
        index, passages, metadata = build_faiss_index()

        # 2️⃣ Hae relevantit kappaleet kysymykseen
        top_passages = retrieve_passages(question, index, passages)

        if not top_passages:
            print("⚠️ Ei relevantteja lähteitä — palautetaan vakio vastaus.")
            return {"answer": "En löydä varmaa ohjetta annetuista lähteistä.", "status": "ok"}

        # 3️⃣ Generoi vastaus mallilla
        answer = generate_answer(question, top_passages)

        # 4️⃣ Vapauta GPU- ja muistiresurssit
        clear_memory()

        return {"answer": answer, "status": "success"}

    except Exception as e:
        clear_memory()
        raise HTTPException(status_code=500, detail=str(e))
