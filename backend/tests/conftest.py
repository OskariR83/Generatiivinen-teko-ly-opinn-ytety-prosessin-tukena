# backend/tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine
from app.main import app


@pytest.fixture(scope="function")
def client(monkeypatch):
    """
    Luo testiasiakas FastAPI:lle ja korvaa tietokantamoottorin
    in-memory SQLite-versiolla.
    """

    # Luo in-memory DB testejä varten
    test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})

    # Korvataan sovelluksen engine testimoottorilla
    from app.database import connection
    monkeypatch.setattr(connection, "engine", test_engine)

    # Luodaan DB-schema
    SQLModel.metadata.create_all(test_engine)

    with TestClient(app) as c:
        yield c
