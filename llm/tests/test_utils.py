"""
test_utils.py
--------------
Yksikkötestit apufunktioille, kuten muistin vapautukselle.
"""

import torch
from llm.src.utils import clear_memory


def test_clear_memory_executes_without_error(monkeypatch):
    """Testaa, että clear_memory suorittuu ilman virheitä."""
    called = {"cuda": False, "gc": False}

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: called.update({"cuda": True}))
    monkeypatch.setattr("gc.collect", lambda: called.update({"gc": True}))

    clear_memory()
    assert called["cuda"]
    assert called["gc"]
