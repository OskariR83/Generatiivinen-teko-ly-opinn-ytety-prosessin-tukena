# llm/pipeline/tests/test_infer_qlora.py

from unittest.mock import patch, MagicMock

@patch("llm.pipeline.infer_qlora.AutoModelForCausalLM")
@patch("llm.pipeline.infer_qlora.AutoTokenizer")
def test_infer_basic(tok_mock, model_mock):
    tok = MagicMock()
    tok_mock.from_pretrained.return_value = tok
    model = MagicMock()
    model_mock.from_pretrained.return_value = model

    from llm.pipeline.infer_qlora import infer_answer

    ans = infer_answer(
        "test question",
        "base_model",
        "adapter_path"
    )

    assert isinstance(ans, str)
