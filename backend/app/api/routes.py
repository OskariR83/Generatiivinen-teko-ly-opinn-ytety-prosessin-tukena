"""
routes.py
----------
Koontireititys kaikille backendin API-päätepisteille:
- /api/test
- /api/feedback (CRUD)
- /api/llm/query
- /api/docs/preprocess, /api/docs/status
"""

from fastapi import APIRouter, HTTPException, Query
from sqlmodel import Session, select
from app.database.connection import engine
from app.models.feedback import Feedback

# Alustetaan päärouter
router = APIRouter(prefix="/api")

# --- TESTI ---
@router.get("/test")
def test_api():
    """Tarkistaa, että API toimii."""
    return {"status": "OK", "message": "API pyörii ja yhteys toimii backendin kautta."}


# --- PALAUTEREITIT ---
@router.post("/feedback", response_model=Feedback)
def create_feedback(message: str = Query(..., description="Käyttäjän palauteviesti")):
    """Tallentaa palautteen tietokantaan."""
    try:
        with Session(engine) as session:
            feedback = Feedback(message=message)
            session.add(feedback)
            session.commit()
            session.refresh(feedback)
            return feedback
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tietokantavirhe: {str(e)}")


@router.get("/feedback")
def list_feedback():
    """Palauttaa kaikki tallennetut palautteet."""
    with Session(engine) as session:
        feedbacks = session.exec(select(Feedback)).all()
        return feedbacks


# --- LLM- JA DOKUMENTTIREITIT ---
# Nämä importit tulevat tiedostoista: app/api/llm_routes.py ja app/api/docs_routes.py
from app.api import llm_routes, docs_routes

router.include_router(llm_routes.router)
router.include_router(docs_routes.router)
