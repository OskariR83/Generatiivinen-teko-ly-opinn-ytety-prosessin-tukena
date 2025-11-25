# llm/tests/test_ocr_utils.py

import pytest
from unittest.mock import MagicMock, patch

@patch("llm.src.ocr_utils.PaddleOCR")
def test_ocr_mock(ocr_mock):
    inst = MagicMock()
    inst.ocr.return_value = [["dummy text"]]
    ocr_mock.return_value = inst

    from llm.src.ocr_utils import ocr_image

    txt = ocr_image("fake_path.png")
    assert "dummy" in txt
