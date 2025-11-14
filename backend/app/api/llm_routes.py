"""
llm_routes.py
-------------
FastAPI endpoint for LLM-based RAG pipeline.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sys
from pathlib import Path

# -------------------------------------------------------------
# Lisää LLM src polku
# -------------------------------------------------------------
BASE_PATH = Path(__file__).resolve().parents[3]
LLM_SRC = BASE_PATH / "llm" / "src"
if str(LLM_SRC) not in sys.path:
    sys.path.insert(0, str(LLM_SRC))

# RAG pipeline modulit
from indexing import build_faiss_index
from retrieval import retrieve_passages
from generation import generate_answer
from utils import clear_memory

router = APIRouter(prefix="/llm", tags=["LLM"])

class LLMQuery(BaseModel):
    question: str

# -------------------------------------------------------------
# 🧠 FAISS-indeksin välimuisti
# -------------------------------------------------------------
INDEX_CACHE = None

def get_index():
    """Rakentaa FAISS-indeksin vain kerran."""
    global INDEX_CACHE
    if INDEX_CACHE is None:
        print("🚀 Rakennetaan FAISS-indeksi ensimmäistä kertaa...")
        INDEX_CACHE = build_faiss_index()
    return INDEX_CACHE

# -------------------------------------------------------------
# 💬 /api/llm/query endpoint
# -------------------------------------------------------------
@router.post("/query")
async def query_llm(data: LLMQuery):
    question = data.question.strip()
    print(f"\n🔎 New query received: {question}\n")
    
    try:
        # 1) Hae tai rakenna FAISS-indeksi vain kerran
        index, passages, metadata = get_index()
        
        # 2) Hae konteksti
        top_passages = retrieve_passages(question, index, passages)
        
        if not top_passages:
            return {
                "answer": "En löydä varmaa ohjetta annetuista lähteistä.", 
                "status": "ok"
            }
        
        # 🔍 Debug: tulosta top-passage -sisältö
        print("\n📄--- TOP PASSAGES (FULL) ---")
        for i, p in enumerate(top_passages, start=1):
            print(f"\n[{i}] {p}\n")
        print("📄--------------------\n")
        
        # 3) Generoi vastaus mallilla (sisältää semantic match checkin)
        answer = generate_answer(question, top_passages)
        
        clear_memory()
        
        return {
            "answer": answer, 
            "status": "success"
        }
        
    except Exception as e:
        clear_memory()
        raise HTTPException(status_code=500, detail=str(e))