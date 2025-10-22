from sqlmodel import SQLModel, create_engine
import os
from dotenv import load_dotenv

load_dotenv()  # lukee .env, jossa DATABASE_URL
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")

engine = create_engine(DATABASE_URL, echo=True)

def init_db():
    SQLModel.metadata.create_all(engine)