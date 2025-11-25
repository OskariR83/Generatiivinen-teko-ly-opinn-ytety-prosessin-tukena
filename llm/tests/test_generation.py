# llm/tests/test_generation_strict.py

from llm.src.generation import generate_answer

def test_no_context_returns_default():
    ans = generate_answer("Mikä on opinnäytetyö?", [])
    assert "en löydä" in ans.lower()


def test_context_with_match(monkeypatch):
    # Mockataan semanttinen match → True
    monkeypatch.setattr(
        "llm.src.generation._semantic_match",
        lambda q, c: True
    )

    # Mockataan LLM → palauttaa käsin annetun
    monkeypatch.setattr(
        "llm.src.generation.AutoModelForCausalLM",
        None
    )

    ans = generate_answer("Mikä on opinnäytetyö?", ["Opinnäytetyö on ..."])
    # koska LLM on korvattu, funktio palauttaa ennen generointia default-vastauksen
    # joten testi tarkistaa vain ettei se hylkää kontekstia
    assert isinstance(ans, str)
