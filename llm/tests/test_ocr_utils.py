"""
test_ocr_utils.py
-----------------
Yksikkötestit Docling- ja EasyOCR-käsittelylle.
"""

import pytest
import json
from pathlib import Path
from llm.src.ocr_utils import process_with_docling, log_ocr_warning, run_easyocr_fallback
from llm.src.config import PROCESSED_DIR, LOG_DIR


def test_log_ocr_warning_creates_file(tmp_path):
    """Testaa, että OCR-varoitus luodaan ja kirjoitetaan lokitiedostoon."""
    test_log = tmp_path / "test_log.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_ocr_warning("dummy.pdf", "Testiviesti")
    log_file = LOG_DIR / "ocr_failures.log"

    assert log_file.exists()
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Testiviesti" in content


def test_process_with_docling_returns_chunks(monkeypatch, tmp_path):
    """Testaa, että Docling palauttaa tekstiä ja jakaa sen kappaleisiin."""
    dummy_file = tmp_path / "dummy.txt"
    dummy_file.write_text("Tämä on testidokumentti, jossa on tekstiä.")

    # Mockataan DocumentConverter palauttamaan testidataa
    class DummyConverter:
        def convert(self, path):
            class DummyDoc:
                def export_to_markdown(self_inner):
                    return "Tämä on testiteksti jota käytetään yksikkötestissä."
            return type("DummyResult", (), {"document": DummyDoc()})()

    monkeypatch.setattr("llm.src.ocr_utils.DocumentConverter", DummyConverter)

    chunks = process_with_docling(str(dummy_file))
    assert len(chunks) > 0
    assert isinstance(chunks[0], str)


def test_run_easyocr_fallback_handles_missing_file(monkeypatch):
    """Testaa, että EasyOCR ei kaadu puuttuessa PDF:ää."""
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)
    result = run_easyocr_fallback("nonexistent.pdf")
    assert isinstance(result, str)
