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
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    poppler-utils \
    git wget curl \
    libjpeg-dev zlib1g-dev

# ================================
# 2️⃣ Luo ja aktivoi virtuaaliympäristö
# ================================
echo "🐍 Luodaan Python-virtuaaliympäristö (venv)..."
python3 -m venv llm/venv
source llm/venv/bin/activate

# ================================
# 3️⃣ Päivitä pip ja asenna Python-kirjastot
# ================================
echo "📚 Asennetaan Python-kirjastot requirements.txt-tiedostosta..."
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

# ================================
# 4️⃣ Asenna PaddlePaddle GPU- tai CPU-versiona
# ================================
echo "🔍 Tarkistetaan CUDA-tuki (GPU-versio PaddleOCR:lle)..."

if python3 - << 'EOF'
import torch
import sys
sys.exit(0 if torch.cuda.is_available() else 1)
EOF
then
    echo "✅ CUDA löytyi — asennetaan PaddlePaddle GPU-versio"
    pip install paddlepaddle-gpu==2.6.1 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
else
    echo "⚠️ CUDA ei käytettävissä — asennetaan PaddlePaddle CPU-versio"
    pip install paddlepaddle==2.6.1
fi

echo "📦 Asennetaan PaddleOCR..."
pip install paddleocr

# ================================
# 5️⃣ Tarkistetaan tärkeimmät kirjastot
# ================================
echo "🔍 Tarkistetaan, että keskeiset paketit toimivat..."

python3 - << 'PYCODE'
import importlib

paketit = [
    "faiss",
    "torch",
    "transformers",
    "sentence_transformers",
    "pymupdf",
    "unstructured",
    "paddleocr"
]

for pkg in paketit:
    try:
        importlib.import_module(pkg)
        print(f"✅ {pkg} asennettu ja toimii")
    except ImportError:
        print(f"❌ {pkg} puuttuu – tarkista asennus!")
PYCODE


# ================================
# 6️⃣ Luo projektin kansiorakenne
# ================================
echo "📁 Luodaan projektin kansiorakenne..."

mkdir -p docs/originals
mkdir -p docs/processed
mkdir -p docs/indexes
mkdir -p logs

echo "✅ Hakemistot luotu."

# ================================
# ✅ Valmis!
# ================================
echo ""
echo "✅ Asennus valmis!"
echo "----------------------------------------------"
echo "Aktivoi virtuaaliympäristö ennen ajoa komennolla:"
echo "  source llm/venv/bin/activate"
echo ""
echo "Aja ohjelma näin:"
echo "  python llm/src/main.py"
echo "----------------------------------------------"
