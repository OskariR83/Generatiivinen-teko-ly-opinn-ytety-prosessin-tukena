# llm/pipeline/tests/test_config_loading.py

from llm.pipeline.train_qlora import load_config

def test_load_config(tmp_path):
    file = tmp_path / "cfg.json"
    file.write_text('{"base_model": "test", "train_path": "x"}')

    cfg = load_config(str(file))
    assert cfg["base_model"] == "test"
