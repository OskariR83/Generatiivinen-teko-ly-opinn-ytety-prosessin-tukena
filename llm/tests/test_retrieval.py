"""
test_retrieval.py
-----------------
Yksikkötestit DPR-kysely- ja hakuosuudelle.
"""

import numpy as np
import torch
from llm.src.retrieval import retrieve_passages


class DummyIndex:
    """Yksinkertainen FAISS-indeksin mock-olio testaukseen."""
    def search(self, q_emb, k):
        # Palautetaan kaksi osumaa testin vuoksi
        return np.array([[0.99, 0.5]]), np.array([[0, 1]])


def test_retrieve_passages_returns_valid_list(monkeypatch):
    """Testaa, että haku palauttaa järkevän tuloksen."""

    # Mockataan tokenizer — palauttaa simuloidun tokeniser-tuloksen
    monkeypatch.setattr(
        "llm.src.retrieval.AutoTokenizer.from_pretrained",
        lambda x: type(
            "T", (), {"__call__": lambda self, q, **kwargs: {"input_ids": [[1, 2, 3]]}}
        )()
    )

    # Mockataan DPRQuestionEncoder — palauttaa tensorin, jolla on .detach()
    monkeypatch.setattr(
        "llm.src.retrieval.DPRQuestionEncoder.from_pretrained",
        lambda x: type(
            "E",
            (),
            {
                "eval": lambda self: None,
                "__call__": lambda self, **kwargs: type(
                    "O", (), {"pooler_output": torch.tensor([[0.1, 0.9]])}
                )(),
            },
        )(),
    )

    dummy_index = DummyIndex()
    passages = ["Ensimmäinen kappale", "Toinen kappale"]

    results = retrieve_passages("testikysymys", dummy_index, passages, k=2)

    # Varmistetaan, että palautetaan järkevä lista tekstikappaleita
    assert isinstance(results, list)
    assert all(isinstance(r, str) for r in results)
    assert len(results) <= 2
