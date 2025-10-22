from fastapi import FastAPI
from app.api import routes
from app.database.connection import init_db

app = FastAPI(
    title="Opinnäytetyön apuri - LLM",
    description="Paikallisesti ajettava kielimalli opinnäytetyöprosessin tueksi.",
    version="0.1"
)


@app.on_event("startup")
def on_startup():
    init_db()
# Reitit
app.include_router(routes.router)

@app.get("/")
def read_root():
    return {"message": "Paikallisesti ajettava kielimalli - Testi - Backend toimii!"}
