# backend/tests/test_llm_routes.py
from unittest.mock import patch
from app.api.llm_routes import SESSION_HISTORY


def test_llm_query_basic(client):
    """Testataan, että /llm/query toimii ilman oikeaa mallia."""

    # Tyhjennä sessiomuisti ennen testiä
    SESSION_HISTORY.clear()

    fake_passages = ["fake passage"]
    fake_answer = "Tämä on testivastaus."

    with patch("app.api.llm_routes.get_or_build_index", return_value=(None, ["dummy"], None)):
        with patch("app.api.llm_routes.retrieve_passages", return_value=fake_passages):
            with patch("app.api.llm_routes.generate_answer", return_value=fake_answer):

                payload = {
                    "question": "Mikä on opinnäytetyö?",
                    "session_id": "test-session-123"
                }

                response = client.post("/llm/query", json=payload)
                json_data = response.json()

                assert response.status_code == 200
                assert json_data["answer"] == fake_answer
                assert json_data["status"] == "success"
                assert json_data["session_id"] == "test-session-123"

                # varmistetaan että historia tallentuu
                assert "test-session-123" in SESSION_HISTORY
                assert len(SESSION_HISTORY["test-session-123"]) == 1


def test_llm_reset(client):
    """Testataan että keskusteluhistoria nollautuu."""

    SESSION_HISTORY["abc"] = [{"user": "hi", "assistant": "hello"}]

    response = client.get("/llm/reset?session_id=abc")
    assert response.status_code == 200

    # varmistetaan ettei historiaa ole enää
    assert "abc" not in SESSION_HISTORY
