"""
utils.py
---------
Sisältää apufunktioita, kuten muistin tyhjennys.
"""

import torch
import gc

def clear_memory():
    """Tyhjentää GPU- ja CPU-muistin."""
    torch.cuda.empty_cache()
    gc.collect()
    print("🧹 GPU- ja muistiresurssit vapautettu.")
