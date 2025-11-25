# llm/pipeline/tests/test_tokenization.py

from transformers import AutoTokenizer
from llm.pipeline.train_qlora import tokenize

def test_tokenizer_mapping():
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    example = {"prompt": "Hello", "labels": "World"}
    f = tokenize(tok, 32)
    out = f(example)

    assert "input_ids" in out
    assert "labels" in out
    assert len(out["input_ids"]) == 32
