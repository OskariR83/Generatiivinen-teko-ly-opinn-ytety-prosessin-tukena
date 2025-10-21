#!/bin/bash
set -e

echo "=== Setting up LLM environment (Doclin + Viking) ==="

# Create venv if not existing
python3 -m venv venv
source venv/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt

pip install "doclin[cuda]"

echo "✅ LLM setup complete!"
