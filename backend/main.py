from fastapi import FastAPI
from app.api.routes import router as api_router
from app.database.connection import init_db
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Opinnäytetyön apuri - LLM",
    description="Paikallisesti ajettava kielimalli opinnäytetyöprosessin tueksi.",
    version="0.1"
)


# ✅ Salli yhteydet frontendiltä
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
