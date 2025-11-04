from sqlmodel import SQLModel, create_engine
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")
engine = create_engine(DATABASE_URL, echo=True)

def init_db():
    """Alustaa tietokannan ja luo taulut."""
    SQLModel.metadata.create_all(engine)
