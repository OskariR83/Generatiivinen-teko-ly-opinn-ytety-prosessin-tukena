from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/llm", tags=["LLM"])


# Pydantic-malli kyselylle
class LLMQuery(BaseModel):
    question: str


# Placeholder-funktio, jossa myöhemmin kutsutaan Tritonia
@router.post("/query")
async def query_llm(data: LLMQuery):
    try:
        # Tähän myöhemmin LLM-kutsu (esim. Triton, RAG, LangChain)
        response_text = f"Placeholder-vastaus kysymykseen: '{data.question}'"

        return {"answer": response_text, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
