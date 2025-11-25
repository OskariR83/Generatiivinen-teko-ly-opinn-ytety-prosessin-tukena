# llm/tests/test_retrieval.py

import pytest
import numpy as np
from llm.src.retrieval import retrieve_passages

@pytest.fixture
def dummy_index(tmp_path):
    dim = 384
    from faiss import IndexFlatL2

    index = IndexFlatL2(dim)
    vectors = np.random.rand(10, dim).astype("float32")
    index.add(vectors)
    return index, ["doc"] * 10


def test_retrieval_basic(dummy_index):
    index, meta = dummy_index
    question = "Miten aloitan opinnäytetyön?"
    results = retrieve_passages(question, index, meta, top_k=3)
    
    assert len(results) == 3
    assert all(isinstance(r, str) for r in results)
