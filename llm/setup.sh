#!/usr/bin/env bash
set -e

echo "🚀 Aloitetaan ympäristön asennus projektille: GENERATIIVINEN TEKOÄLY OPINNÄYTETYÖPROSESSIN TUKENA"

# ================================
# 1️⃣ Päivitä järjestelmä ja asenna tarvittavat kirjastot
# ================================
echo "📦 Asennetaan järjestelmätason riippuvuudet..."
sudo apt update -y
sudo apt install -y \
    python3 python3-venv python3-pip \
    build-essential \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    git wget curl

# ================================
# 2️⃣ Luo ja aktivoi virtuaaliympäristö
# ================================
echo "🐍 Luodaan Python-virtuaaliympäristö (venv)..."
python3 -m venv llm/venv
source llm/venv/bin/activate

# ================================
# 3️⃣ Asennetaan Python-paketit
# ================================
echo "📚 Asennetaan Python-kirjastot requirements.txt-tiedostosta..."
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

# ================================
# 4️⃣ Tarkistetaan OCR-komponentit ja tärkeimmät kirjastot
# ================================
echo "🔍 Tarkistetaan, että tärkeimmät kirjastot ovat käytettävissä..."
python3 - <<'PYCODE'
import importlib
paketit = [
    "docling",
    "rapidocr_onnxruntime",
    "easyocr",
    "pdf2image",
    "faiss",
    "torch",
    "transformers"
]
for pkg in paketit:
    try:
        importlib.import_module(pkg)
        print(f"✅ {pkg} asennettu ja toimii")
    except ImportError:
        print(f"⚠️ {pkg} puuttuu – tarkista asennus.")
PYCODE

# ================================
# 5️⃣ Valmis!
# ================================
echo ""
echo "✅ Asennus valmis!"
echo "----------------------------------------------"
echo "Aktivoi virtuaaliympäristö ennen ajoa komennolla:"
echo "  source llm/venv/bin/activate"
echo ""
echo "Aja ohjelma näin:"
echo "  python llm/src/rag_pipeline.py"
echo "----------------------------------------------"
