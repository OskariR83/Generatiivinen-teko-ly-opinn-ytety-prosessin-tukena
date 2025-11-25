# llm/pipeline/tests/test_prompt_builder.py

from llm.pipeline.train_qlora import make_prompt

def test_make_prompt_basic():
    ex = {
        "instruction": "Miten aloitan opinnäytetyön?",
        "input": "",
        "output": "Aloita valitsemalla aihe."
    }

    r = make_prompt(ex)
    assert "Kysymys" in r["prompt"]
    assert "Aloita" in r["labels"]
