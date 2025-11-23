"""
llm_routes.py
-------------
FastAPI endpoints for LLM-based RAG pipeline with session-based conversation history.

Toteutus:
- FAISS-indeksin lazy-initialisointi
- Strict retrieval (TurkuNLP + FAISS)
- Strict generation (Viking-7B), johon lisätään kevyt keskusteluhistoria
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import sys
from pathlib import Path

# -------------------------------------------------------------
# LLM-moduulin polku
# -------------------------------------------------------------
BASE_PATH = Path(__file__).resolve().parents[3]  
LLM_SRC = BASE_PATH / "llm" / "src"
if str(LLM_SRC) not in sys.path:
    sys.path.insert(0, str(LLM_SRC))

# RAG-komponentit
from retrieval import retrieve_passages
from generation import generate_answer
from indexing import build_faiss_index
from utils import clear_memory

router = APIRouter(prefix="/llm", tags=["LLM"])


# -------------------------------------------------------------
# Pydantic-malli pyynnölle
# -------------------------------------------------------------
class LLMQuery(BaseModel):
    question: str
    session_id: str


# -------------------------------------------------------------
# Session-muisti (RAM)
# -------------------------------------------------------------
# Rakenne:
# SESSION_HISTORY = {
#   "session_id": [
#       {"user": "kysymys", "assistant": "vastaus"},
#       ...
#   ]
# }
SESSION_HISTORY: dict[str, list[dict[str, str]]] = {}


# -------------------------------------------------------------
# FAISS-indeksin lazy-initialisointi
# -------------------------------------------------------------
INDEX_DATA = None  # cache: (index, passages, metadata)


def get_or_build_index():
    """Rakentaa FAISS-indeksin vain kerran, ensimmäisellä kutsulla."""
    global INDEX_DATA
    if INDEX_DATA is None:
        print("🚀 Rakennetaan FAISS-indeksi ensimmäistä kertaa...")
        INDEX_DATA = build_faiss_index()
        if INDEX_DATA is None:
            raise RuntimeError("FAISS-indeksin rakentaminen epäonnistui. Tarkista prosessoidut dokumentit.")
        print("✅ FAISS-indeksi valmis.")
    return INDEX_DATA


# -------------------------------------------------------------
# Keskustelun nollaus
# -------------------------------------------------------------
@router.get("/reset")
async def reset_session(session_id: str = Query(..., description="Frontendin generoima sessionId")):
    """
    Tyhjentää annetun session keskusteluhistorian RAM-muistista.
    Ei koske dokumentti-indeksejä tai tietokantaa.
    """
    if session_id in SESSION_HISTORY:
        del SESSION_HISTORY[session_id]
        return {"status": "ok", "message": "Keskusteluhistoria tyhjennetty."}
    return {"status": "ok", "message": "Sessiossa ei ollut olemassa olevaa historiaa."}


# -------------------------------------------------------------
# Apufunktio: muodosta kevyt keskusteluhistoria generointia varten
# -------------------------------------------------------------
def build_history_context(session_id: str, max_turns: int = 5) -> str:
    """
    Rakentaa lyhyen tekstimuotoisen historian generointia varten.
    Historiaa EI käytetä retrievalissa, ainoastaan LLM:n kontekstina.
    """
    turns = SESSION_HISTORY.get(session_id, [])
    if not turns:
        return ""

    recent = turns[-max_turns:]
    lines = []
    for turn in recent:
        user_q = turn.get("user", "").strip()
        assistant_a = turn.get("assistant", "").strip()
        if user_q:
            lines.append(f"User: {user_q}")
        if assistant_a:
            lines.append(f"Assistant: {assistant_a}")
        lines.append("")  # tyhjä rivi väliin

    history_text = "\n".join(lines).strip()
    return history_text


# -------------------------------------------------------------
# Pääkysely /llm/query
# -------------------------------------------------------------
@router.post("/query")
async def query_llm(data: LLMQuery):
    """
    Suorittaa RAG-pipeline-kyselyn:

    1) käyttää VAIN tämänhetkistä kysymystä FAISS-hakuun
    2) generointiin lisätään dokumenttikontekstin lisäksi kevyt keskusteluhistoria
    3) vastaus palautetaan JSON-muodossa
    """
    question = data.question.strip()
    session_id = data.session_id

    print(f"\n🔎 New query from session {session_id}: {question}\n")

    # Alusta session-historia tarvittaessa
    if session_id not in SESSION_HISTORY:
        SESSION_HISTORY[session_id] = []

    try:
        # 1) Lataa tai rakenna FAISS-indeksi (vain kerran)
        index, passages, metadata = get_or_build_index()

        # 2) STRICT RETRIEVAL: käytetään VAIN nykyistä kysymystä
        top_passages = retrieve_passages(question, index, passages)

        print("\n📄--- TOP PASSAGES (FULL) ---")
        for i, p in enumerate(top_passages, start=1):
            print(f"\n[{i}] {p}\n")
        print("📄--------------------\n")

        if not top_passages:
            answer = "En löydä varmaa ohjetta annetuista lähteistä."
        else:
            # 3) Muodosta kevyt keskusteluhistoria generointia varten
            history_text = build_history_context(session_id)

            # 3a) Lisätään historia generoinnin kontekstiin ERILLISENÄ kappaleena
            #     → retrieval pysyy puhtaana, mutta LLM näkee aiemman keskustelun
            generation_context = list(top_passages)
            if history_text:
                history_passage = (
                    "Keskusteluhistoria (älä keksi uutta tietoa, käytä silti vain "
                    "dokumenttikontekstia faktatietoon):\n\n" + history_text
                )
                generation_context.insert(0, history_passage)

            # 4) GENEROINTI:
            #    - question: nykyinen kysymys
            #    - generation_context: dokumenttikonteksti + kevyt historia
            answer = generate_answer(question, generation_context)

        # 5) Tallenna kysymys–vastaus session-muistiin
        SESSION_HISTORY[session_id].append(
            {
                "user": question,
                "assistant": answer,
            }
        )

        clear_memory()

        return {
            "answer": answer,
            "status": "success",
            "session_id": session_id,
        }

    except Exception as e:
        clear_memory()
        raise HTTPException(status_code=500, detail=str(e))
