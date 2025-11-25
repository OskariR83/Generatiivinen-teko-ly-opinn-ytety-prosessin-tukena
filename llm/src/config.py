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

# ============================
# QLoRA-integraatio RAG-pipelineen
# ============================

# Käytetäänkö QLoRA-adaptereita vai pelkkää perusmallia?
USE_QLORA = True   # jos haluat palata alkuperäiseen Vikingiin, vaihda False

# Perusmalli (sama kuin QLoRA-koulutuksessa)
QLORA_BASE_MODEL = "mpasila/Alpacazord-Viking-7B"

# Polku LoRA-adaptereihin (output_dir config_qlora_viking7b.json:issa)
QLORA_ADAPTER_PATH = "llm/pipeline/output/viking7b-qlora-ont"
