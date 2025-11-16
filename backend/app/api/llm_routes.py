"""
llm_routes.py
-------------
FastAPI endpoints for LLM-based RAG pipeline with session-based conversation history.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import sys
from pathlib import Path

# -------------------------------------------------------------
# Add LLM module path
# -------------------------------------------------------------
BASE_PATH = Path(__file__).resolve().parents[3]
LLM_SRC = BASE_PATH / "llm" / "src"
if str(LLM_SRC) not in sys.path:
    sys.path.insert(0, str(LLM_SRC))

# Import components
from retrieval import retrieve_passages
from generation import generate_answer
from indexing import build_faiss_index
from utils import clear_memory

router = APIRouter(prefix="/llm", tags=["LLM"])


# -------------------------------------------------------------
# Request schema
# -------------------------------------------------------------
class LLMQuery(BaseModel):
    question: str
    session_id: str


# -------------------------------------------------------------
# Session memory (RAM)
# -------------------------------------------------------------
SESSION_HISTORY = {}


# -------------------------------------------------------------
# Lazy index init
# -------------------------------------------------------------
INDEX_DATA = None

def get_or_build_index():
    global INDEX_DATA
    if INDEX_DATA is None:
        print("🚀 Building FAISS index first time...")
        INDEX_DATA = build_faiss_index()
    return INDEX_DATA


# -------------------------------------------------------------
# Reset endpoint
# -------------------------------------------------------------
@router.get("/reset")
async def reset_session(session_id: str = Query(...)):
    if session_id in SESSION_HISTORY:
        del SESSION_HISTORY[session_id]
    return {"status": "ok", "message": "session cleared"}


# -------------------------------------------------------------
# Main query endpoint
# -------------------------------------------------------------
@router.post("/query")
async def query_llm(data: LLMQuery):
    question = data.question.strip()
    session_id = data.session_id

    print(f"\n🔎 New query from session {session_id}: {question}\n")

    # Initialize session memory if needed
    if session_id not in SESSION_HISTORY:
        SESSION_HISTORY[session_id] = []

    try:
        # 1) Load or build FAISS index
        index, passages, metadata = get_or_build_index()

        # 2) Build conversational context
        history_context = ""
        for turn in SESSION_HISTORY[session_id][-5:]:
            history_context += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

        full_query = history_context + f"User: {question}"

        # 3) Retrieval
        top_passages = retrieve_passages(full_query, index, passages)

        # Debug print
        print("\n📄--- TOP PASSAGES (FULL) ---")
        for i, p in enumerate(top_passages, start=1):
            print(f"\n[{i}] {p}\n")
        print("📄--------------------\n")

        if not top_passages:
            answer = "En löydä varmaa ohjetta annetuista lähteistä."
        else:
            # 4) Generate
            answer = generate_answer(question, top_passages)

        # 5) Store Q/A to memory
        SESSION_HISTORY[session_id].append({
            "user": question,
            "assistant": answer
        })

        clear_memory()

        return {"answer": answer, "status": "success", "session_id": session_id}

    except Exception as e:
        clear_memory()
        raise HTTPException(status_code=500, detail=str(e))
