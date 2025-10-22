from fastapi import APIRouter, HTTPException
from sqlmodel import Session, select
from app.database.connection import engine
from app.models.feedback import Feedback

router = APIRouter(prefix="/api")

@router.get("/test")
def test_api():
    return {"status": "OK", "message": "API pyörii ja yhteys toimii backendin kautta."}

@router.post("/feedback", response_model=Feedback)
def create_feedback(message: str):
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
    with Session(engine) as session:
        feedbacks = session.exec(select(Feedback)).all()
        return feedbacks
