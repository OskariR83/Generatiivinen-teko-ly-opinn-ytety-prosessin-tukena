"""
test_generation.py
------------------
Yksikkötestit Viking-7B -mallin vastausgeneroinnille.
"""

import torch
from llm.src.generation import generate_answer


def test_generate_answer_returns_string(monkeypatch):
    """Testaa, että generointi palauttaa tekstin ilman virhettä."""

    # Mockataan tokenizer ja model kevyesti
    class DummyTokenizer:
        def __init__(self):
            self.pad_token = None
            self.eos_token = ""

        def encode(self, x):
            return [1, 2, 3]

        def __call__(self, text, **kwargs):
            # Palautetaan olio, jolla on .to() ja .input_ids attribuutti (tensorina)
            class DummyInputs(dict):
                def __init__(self, data):
                    super().__init__(data)
                    self.input_ids = torch.tensor(data["input_ids"])  # tensori = .shape toimii
                def to(self, device):
                    return self
            return DummyInputs({"input_ids": [[1, 2, 3]]})

        def decode(self, ids, skip_special_tokens=True):
            return "Tämä on testivastaus."

    class DummyModel:
        def __init__(self):
            self.device = "cpu"

        def eval(self):
            pass

        def generate(self, **kwargs):
            # Palautetaan feikki token-lista
            return [[1, 2, 3, 4, 5, 6]]

    # Korvataan oikeat mallit mockeilla
    monkeypatch.setattr("llm.src.generation.AutoTokenizer.from_pretrained", lambda *a, **k: DummyTokenizer())
    monkeypatch.setattr("llm.src.generation.AutoModelForCausalLM.from_pretrained", lambda *a, **k: DummyModel())

    # Suoritetaan testi
    result = generate_answer("Testikysymys", ["Tämä on testikappale."])

    # Varmistetaan, että palautetaan teksti
    assert isinstance(result, str)
    assert "testivastaus" in result.lower()
