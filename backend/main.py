from fastapi import FastAPI
from app.api.routes import router as api_router
from app.database.connection import init_db

app = FastAPI(
    title="Opinnäytetyön apuri - LLM",
    description="Paikallisesti ajettava kielimalli opinnäytetyöprosessin tueksi.",
    version="0.1"
)


@app.on_event("startup")
def on_startup():
    """Tietokannan alustaminen sovelluksen käynnistyessä."""
    init_db()


# --- API-reitit ---
app.include_router(api_router)


@app.get("/")
def read_root():
    """Tarkistus juurireitillä."""
    return {"message": "Paikallisesti ajettava kielimalli - Backend toimii!"}
