"""
config.py
----------
Sisältää projektin asetukset, hakemistopolut ja OCR-asetukset.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DOCS_DIR = BASE_DIR / "docs"
ORIGINALS_DIR = DOCS_DIR / "originals"
PROCESSED_DIR = DOCS_DIR / "processed"
INDEX_DIR = DOCS_DIR / "indexes"
LOG_DIR = BASE_DIR / "logs"

# Luodaan tarvittavat hakemistot
for d in [DOCS_DIR, ORIGINALS_DIR, PROCESSED_DIR, INDEX_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# OCR-asetukset
FORCE_EASYOCR = False  # True = käytä aina EasyOCR
