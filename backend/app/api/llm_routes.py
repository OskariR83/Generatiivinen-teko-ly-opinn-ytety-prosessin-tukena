"""
llm_routes.py
-------------
FastAPI:n päätepisteet kielimallipohjaiseen RAG-hakuun.
Sisältää myös session-pohjaisen keskusteluhistorian tallennuksen.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import sys
from pathlib import Path

# -------------------------------------------------------------
# LLM-moduulin polku projektin juureen asti
# -------------------------------------------------------------
BASE_PATH = Path(__file__).resolve().parents[3]
LLM_SRC = BASE_PATH / "llm" / "src"
if str(LLM_SRC) not in sys.path:
    sys.path.insert(0, str(LLM_SRC))

# -------------------------------------------------------------
# RAG-komponentit
# -------------------------------------------------------------
from retrieval import retrieve_passages
from generation import generate_answer
from indexing import build_faiss_index
from utils import clear_memory

# FastAPI-router
router = APIRouter(prefix="/llm", tags=["LLM"])


# -------------------------------------------------------------
# Pydantic-malli
# -------------------------------------------------------------
class LLMQuery(BaseModel):
    question: str
    session_id: str


# -------------------------------------------------------------
# Sessionkohtainen keskusteluhistoria (RAM)
# -------------------------------------------------------------
SESSION_HISTORY = {}


# -------------------------------------------------------------
# FAISS-indeksin lazy-init
# -------------------------------------------------------------
INDEX_DATA = None

def get_or_build_index():
    """Rakentaa FAISS-indeksin vain kerran ensimmäisellä kutsulla."""
    global INDEX_DATA
    if INDEX_DATA is None:
        print("🚀 Rakennetaan FAISS-indeksi ensimmäistä kertaa...")
        INDEX_DATA = build_faiss_index()
    return INDEX_DATA


# -------------------------------------------------------------
# Keskustelun nollaus
# -------------------------------------------------------------
@router.get("/reset")
async def reset_session(session_id: str = Query(..., description="Selaimen session tunniste")):
    """Tyhjentää yksittäisen session keskusteluhistorian."""
    if session_id in SESSION_HISTORY:
        del SESSION_HISTORY[session_id]
    return {"status": "ok", "message": "Keskusteluhistoria tyhjennetty."}


# -------------------------------------------------------------
# Päätepiste: LLM-kysely (RAG + keskusteluhistoria)
# -------------------------------------------------------------
@router.post("/query")
async def query_llm(data: LLMQuery):
    """
    Vastaanottaa käyttäjän kysymyksen ja tuottaa vastauksen
    hyödyntämällä RAG-pipelinea sekä sessionkohtaista keskusteluhistoriaa.
    """
    question = data.question.strip()
    session_id = data.session_id

    print(f"\n🔎 Uusi kysely (session: {session_id}): {question}\n")

    # Luo session-historia tarvittaessa
    if session_id not in SESSION_HISTORY:
        SESSION_HISTORY[session_id] = []

    try:
        # 1) Lataa tai rakenna FAISS-indeksi
        index, passages, metadata = get_or_build_index()

        # 2) Rakenna konteksti viimeisistä viesteistä
        history_context = ""
        for turn in SESSION_HISTORY[session_id][-5:]:
            history_context += (
                f"User: {turn['user']}\n"
                f"Assistant: {turn['assistant']}\n\n"
            )

        full_query = history_context + f"User: {question}"

        # 3) Semanttinen haku
        top_passages = retrieve_passages(full_query, index, passages)

        # Tulosta debug-informaatiota
        print("\n📄--- TOP PASSAGES ---")
        for i, p in enumerate(top_passages, start=1):
            print(f"[{i}] {p}\n")
        print("📄---------------------\n")

        # 4) Generointi
        if not top_passages:
            answer = "En löydä varmaa ohjetta annetuista lähteistä."
        else:
            answer = generate_answer(question, top_passages)

        # 5) Tallenna Q/A muistiin
        SESSION_HISTORY[session_id].append({
            "user": question,
            "assistant": answer
        })

        # 6) Siivoa LLM-muisti
        clear_memory()

        return {
            "answer": answer,
            "status": "success",
            "session_id": session_id
        }

    except Exception as e:
        clear_memory()
        raise HTTPException(status_code=500, detail=str(e))
