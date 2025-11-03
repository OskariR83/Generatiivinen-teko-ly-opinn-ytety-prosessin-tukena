"""
test_indexing.py
----------------
Yksikkötestit FAISS-indeksin rakennukselle ja lataukselle.
"""

import pytest
import json
import numpy as np
from pathlib import Path
from llm.src.indexing import build_faiss_index


def test_build_faiss_index_creates_files(monkeypatch, tmp_path):
    """Testaa, että FAISS-indeksi ja metadata luodaan onnistuneesti."""
    dummy_doc = tmp_path / "test.txt"
    dummy_doc.write_text("Testidokumentti sisältää testisisältöä FAISS-testiin.")

    # Mockataan prosessointi
    monkeypatch.setattr("llm.src.indexing.process_with_docling", lambda x: ["test passage 1", "test passage 2"])

    index_file = tmp_path / "test_index.faiss"
    index, passages, meta = build_faiss_index(base_docs_dir=tmp_path, index_path=index_file)

    assert index_file.exists()
    assert len(passages) == 2
    assert isinstance(meta, list)
    assert hasattr(index, "ntotal")
