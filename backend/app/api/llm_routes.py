"""
llm_routes.py
-------------
FastAPI endpoint for LLM-based RAG pipeline.

This route handles:
1) FAISS index lazy initialization (on first use)
2) Semantic retrieval (v3.5)
3) Strict answer generation (v3.1)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sys
from pathlib import Path

# -------------------------------------------------------------
# 🧩 Add LLM module path
# -------------------------------------------------------------
BASE_PATH = Path(__file__).resolve().parents[3]  # backend/app/api → project root
LLM_SRC = BASE_PATH / "llm" / "src"
if str(LLM_SRC) not in sys.path:
    sys.path.insert(0, str(LLM_SRC))

# -------------------------------------------------------------
# 🧠 Import RAG components
# -------------------------------------------------------------
from retrieval import retrieve_passages
from generation import generate_answer
from indexing import build_faiss_index
from utils import clear_memory

router = APIRouter(prefix="/llm", tags=["LLM"])


# -------------------------------------------------------------
# 📘 Pydantic model
# -------------------------------------------------------------
class LLMQuery(BaseModel):
    question: str


# -------------------------------------------------------------
# 🧱 Lazy index initialization
# -------------------------------------------------------------
INDEX_DATA = None  # cache (index, passages, meta)


def get_or_build_index():
    """Build FAISS index only once, when first needed."""
    global INDEX_DATA
    if INDEX_DATA is None:
        print("🚀 Building FAISS index on first use...")
        INDEX_DATA = build_faiss_index()
        if INDEX_DATA is None:
            raise RuntimeError("Failed to build FAISS index. Check processed docs.")
        print("✅ FAISS index ready.")
    return INDEX_DATA


# -------------------------------------------------------------
# 💬 /api/llm/query endpoint
# -------------------------------------------------------------
@router.post("/query")
async def query_llm(data: LLMQuery):
    """
    Suorittaa RAG-pipelinekyselyn:
    1) hakee relevantit kappaleet FAISS-indeksistä
    2) generoi tiiviin vastauksen Viking-7B -mallilla
    3) palauttaa tuloksen JSON-muodossa
    """
    question = data.question
    print(f"\n🔎 New query received: {question}\n")

    try:
        # 1) Load or build FAISS index
        index, passages, metadata = get_or_build_index()

        # 2) Retrieve relevant passages
        top_passages = retrieve_passages(question, index, passages)
        if not top_passages:
            return {"answer": "En löydä varmaa ohjetta annetuista lähteistä.", "status": "ok"}

        # 3) Generate answer
        answer = generate_answer(question, top_passages)

        # 4) Cleanup
        clear_memory()

        return {"answer": answer, "status": "success"}

    except Exception as e:
        clear_memory()
        raise HTTPException(status_code=500, detail=str(e))
