"""
ocr_utils.py
-------------
Päivitetty versio: käyttää Unstructured.io- ja PyMuPDF-pohjaista dokumenttien
käsittelyä, korvaa vanhan Docling + EasyOCR -pipeline.
Säilyttää dokumentin rakenteen (otsikot, kappaleet, luettelot)
ja toimii täysin offline. Yhteensopiva indexing.py:n kanssa.
"""
import os

import json
from pathlib import Path
from datetime import datetime
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.text import partition_text
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import NarrativeText, Title, ListItem
from .config import PROCESSED_DIR, LOG_DIR
import fitz  # PyMuPDF



# ---------------------------------------------------------------------------
# 🔧 Lokitustuki
# ---------------------------------------------------------------------------
def log_ocr_warning(file_path, message):
    """Kirjaa OCR- ja käsittelyvaroitukset lokitiedostoon."""
    log_file = LOG_DIR / "ocr_failures.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {file_path}: {message}\n")
    print(f"⚠️ {message}")


# ---------------------------------------------------------------------------
# 📘 Dokumenttien käsittely
# ---------------------------------------------------------------------------
def extract_text_unstructured(file_path: str):
    """Lukee PDF-, DOCX- tai TXT-tiedoston ja palauttaa sen tekstin."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    try:
        if suffix == ".pdf":
            elements = partition_pdf(
            file_path,
            strategy="hi_res",
            infer_table_structure=True,
            languages=["fi", "en"],  # ✅ OCR suomi + englanti
        )

        elif suffix == ".docx":
            elements = partition_docx(file_path)
        elif suffix == ".txt":
            elements = partition_text(file_path)
        else:
            log_ocr_warning(file_path, f"❌ Tuntematon tiedostomuoto: {suffix}")
            return ""

        # Poistetaan ei-tekstielementit (kuvat, taulukot, metatiedot)
        text_elements = [
            el for el in elements
            if isinstance(el, (NarrativeText, Title, ListItem))
        ]

        full_text = "\n\n".join([el.text for el in text_elements if el.text.strip()])
        return full_text.strip()

    except Exception as e:
        log_ocr_warning(file_path, f"Unstructured-käsittely epäonnistui: {e}")
        return ""


# ---------------------------------------------------------------------------
# 🧠 OCR-varamenetelmä
# ---------------------------------------------------------------------------
def run_paddleocr_fallback(file_path: str):
    """Käyttää PaddleOCR:ia, jos PDF:ssä ei ole tekstitasoja."""
    try:
        from paddleocr import PaddleOCR
        import torch

        print(f"🔍 Käynnistetään PaddleOCR-varamenetelmä: {file_path}")
        use_gpu = torch.cuda.is_available()  # ✅ tarkistaa, onko GPU käytössä
        ocr = PaddleOCR(lang="fi", use_angle_cls=True, use_gpu=use_gpu)  # ✅ käyttää GPU:ta, jos saatavilla
        doc = fitz.open(file_path)

        text_output = ""
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=200)
            result = ocr.ocr(pix.tobytes(), cls=False)
            page_text = " ".join(
                [line[1][0] for line in result[0]]
            ) if result and result[0] else ""
            print(f"📄 OCR-sivu {i+1}: {len(page_text)} merkkiä")
            text_output += page_text + "\n"

        return text_output.strip()

    except Exception as e:
        log_ocr_warning(file_path, f"PaddleOCR epäonnistui: {e}")
        return ""


# ---------------------------------------------------------------------------
# ⚙️ Prosessointipääfunktio
# ---------------------------------------------------------------------------
def process_with_unstructured(file_path: str):
    """
    Käsittelee dokumentin rakenteisesti Unstructuredin avulla.
    Tallentaa JSON-tiedoston ja palauttaa tekstikappaleet.
    """
    raw_path = Path(file_path)
    output_file = PROCESSED_DIR / f"{raw_path.stem}_clean.json"

    if not raw_path.exists():
        log_ocr_warning(file_path, "❌ Tiedostoa ei löydy.")
        return []

    # Käytä välimuistia jos käsitelty aiemmin
    if output_file.exists():
        print(f"📂 Käytetään välimuistissa olevaa tiedostoa: {output_file}")
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            text = data.get("text", "")
            return text_to_chunks(text)

    print(f"🧠 Prosessoidaan dokumentti Unstructuredilla: {file_path}")
    text_output = extract_text_unstructured(file_path)

    # Jos ei löydetty tekstiä → kokeillaan OCR
    if not text_output.strip():
        log_ocr_warning(file_path, "⚠️ Ei tekstitasoa – käytetään OCR-varamenetelmää.")
        text_output = run_paddleocr_fallback(file_path)

    if not text_output.strip():
        log_ocr_warning(file_path, "❌ OCR epäonnistui kokonaan.")
        return []

    # Tallenna käsitelty dokumentti
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data = {"text": text_output}
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"💾 Tallennettu käsitelty dokumentti: {output_file}")
    return text_to_chunks(text_output)


# ---------------------------------------------------------------------------
# ✂️ Chunking-logiikka
# ---------------------------------------------------------------------------
def text_to_chunks(text: str, chunk_size: int = 400):
    """Jakaa tekstin loogisiksi kappaleiksi säilyttäen otsikkorakenteen."""
    paragraphs = text.split("\n\n")
    chunks, current_chunk = [], ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    print(f"✅ Prosessoidusta dokumentista saatiin {len(chunks)} tekstikappaletta.")
    return chunks
def preprocess_all_documents(originals_dir=None, processed_dir=None):
    """
    Käsittelee kaikki alkuperäiset dokumentit (PDF, DOCX, TXT) kansiosta `docs/originals`
    ja tallentaa ne JSON-muodossa kansioon `docs/processed`.

    Dokumentteja ei käsitellä uudelleen, jos vastaava *_clean.json on jo olemassa.
    """
    from pathlib import Path

    base_dir = Path(__file__).resolve().parents[2]
    originals_dir = Path(originals_dir or (base_dir / "docs/originals"))
    processed_dir = Path(processed_dir or (base_dir / "docs/processed"))

    originals_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    files = [f for f in originals_dir.iterdir() if f.suffix.lower() in [".pdf", ".docx", ".txt"]]
    if not files:
        print(f"⚠️ Ei käsiteltäviä dokumentteja hakemistossa: {originals_dir}")
        return

    print(f"📄 Käsitellään {len(files)} dokumenttia ennen indeksointia...\n")

    for f in files:
        processed_file = processed_dir / f"{f.stem}_clean.json"
        if processed_file.exists():
            print(f"✅ Välimuistissa: {f.name}")
            continue

        try:
            print(f"🧠 Prosessoidaan dokumentti: {f.name}")
            text_chunks = process_with_unstructured(str(f))

            if text_chunks:
                text_output = " ".join(text_chunks)
                data = {"text": text_output}

                with open(processed_file, "w", encoding="utf-8") as outfile:
                    json.dump(data, outfile, ensure_ascii=False, indent=2)

                print(f"💾 Tallennettu: {processed_file}")
            else:
                print(f"⚠️ Ei tekstiä dokumentista: {f.name}")

        except Exception as e:
            log_ocr_warning(str(f), f"Virhe dokumentin käsittelyssä: {e}")
            print(f"❌ Virhe käsiteltäessä {f.name}: {e}")

    print("\n✅ Kaikki dokumentit käsitelty.")
