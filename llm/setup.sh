#!/usr/bin/env bash
set -e

echo "🚀 Aloitetaan toimivan ympäristön asennus"

# ================================
# 1️⃣ Järjestelmätason paketit
# ================================
echo "📦 Asennetaan järjestelmäriippuvuudet..."
sudo apt update -y
sudo apt install -y \
    python3 python3-venv python3-pip \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    poppler-utils \
    git wget curl \
    libjpeg-dev zlib1g-dev

# ================================
# 2️⃣ Luo ja aktivoi venv
# ================================
echo "🐍 Luodaan Python venv..."
python3 -m venv llm/venv
source llm/venv/bin/activate

# ================================
# 3️⃣ Pip + Python-paketit
# ================================
echo "📚 Päivitetään pip ja asennetaan paketit..."
pip install --upgrade pip wheel setuptools

pip install -r requirements_working.txt

# ================================
# 4️⃣ PaddleOCR (vain CPU-tuki)
# ================================
echo "📦 Asennetaan PaddleOCR..."
pip install paddlepaddle==2.6.1
pip install paddleocr==2.7.0.3

# ================================
# 5️⃣ Tarkistetaan keskeiset paketit
# ================================
echo "🔍 Tarkistetaan kirjastot..."

python3 - << 'EOF'
import importlib

paketit = [
    "torch",
    "transformers",
    "sentence_transformers",
    "faiss",
    "unstructured",
    "unstructured_inference",
    "pymupdf",
    "paddleocr"
]

for p in paketit:
    try:
        importlib.import_module(p)
        print(f"✅ {p} OK")
    except:
        print(f"❌ VIRHE: {p} EI toimi!")
EOF

# ================================
# 6️⃣ Projektin kansiot
# ================================
echo "📁 Luodaan projektihakemistot..."

mkdir -p docs/originals
mkdir -p docs/processed
mkdir -p docs/indexes
mkdir -p logs

echo "🎉 Ympäristö valmis!"
echo "--------------------------------------"
echo "Aktivoi ympäristö:"
echo "  source llm/venv/bin/activate"
echo ""
echo "Aja ohjelma:"
echo "  python llm/src/main.py"
echo "--------------------------------------"
