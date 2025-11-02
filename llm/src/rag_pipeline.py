"""
RAG-putki: Docling + DPR + FAISS (CPU) + Viking-7B / Alpacazord-Viking-7B / Viking-13B-GGUF
Projekti: GENERATIIVINEN TEKOÄLY OPINNÄYTETYÖPROSESSIN TUKENA

Kuvaus:
    Tämä ohjelma muodostaa päästä päähän RAG-prosessin, joka:
    - esikäsittelee ja muuntaa dokumentit tekstimuotoon Docling-kirjastolla
    - käyttää OCR:ää (RapidOCR tai EasyOCR)
    - laskee dokumenttikappaleiden upotukset DPR-mallilla
    - tallentaa upotukset FAISS-indeksiin (CPU)
    - hakee parhaiten vastaavat kappaleet ja tuottaa vastauksen Viking-7B-mallilla
"""

import os
import sys
import json
import numpy as np
import faiss
import torch
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DPRContextEncoder,
    DPRQuestionEncoder,
)
from docling.document_converter import DocumentConverter
from transformers.utils import logging
import gc
from tqdm import tqdm

# ===========================================================
# 🔧 Korjataan Doclingin lokitus reaaliaikaiseksi
# ===========================================================
import logging as pylogging

for handler in pylogging.root.handlers[:]:
    pylogging.root.removeHandler(handler)

class FlushStreamHandler(pylogging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

pylogging.basicConfig(
    level=pylogging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[FlushStreamHandler(sys.stdout)],
    force=True
)

# ===========================================================
# Perusasetukset ja kansiot
# ===========================================================
logging.set_verbosity_error()

BASE_DIR = Path(__file__).resolve().parents[2]
DOCS_DIR = BASE_DIR / "docs"
ORIGINALS_DIR = DOCS_DIR / "originals"
PROCESSED_DIR = DOCS_DIR / "processed"
INDEX_DIR = DOCS_DIR / "indexes"
LOG_DIR = BASE_DIR / "logs"

for d in [DOCS_DIR, ORIGINALS_DIR, PROCESSED_DIR, INDEX_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"📂 Projektihakemisto: {BASE_DIR}")
print(f"📄 Alkuperäiset dokumentit: {ORIGINALS_DIR}")
print(f"🧹 Prosessoidut tiedostot: {PROCESSED_DIR}")
print(f"🧠 Indeksit: {INDEX_DIR}")

# OCR-pakotuksen valinta
FORCE_EASYOCR = False  # True = käytä aina EasyOCR:ia

# ===========================================================
# Lokitusapu
# ===========================================================
def kirjaa_ocr_varoitus(file_path, viesti):
    log_file = LOG_DIR / "ocr_failures.log"
    aikaleima = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{aikaleima}] {file_path}: {viesti}\n")
    print(f"⚠️ {viesti}")

# ===========================================================
# EasyOCR-varamenetelmä
# ===========================================================
def tee_easyocr_varamenetelma(file_path: str):
    """Käyttää EasyOCR:ia varamenetelmänä, jos RapidOCR epäonnistuu."""
    try:
        import easyocr
        from pdf2image import convert_from_path
        import numpy as np

        print(f"🔄 Käynnistetään EasyOCR-varamenetelmä: {file_path}")
        reader = easyocr.Reader(["fi", "en"], gpu=torch.cuda.is_available())

        pages = convert_from_path(file_path, dpi=200)
        text_output = ""

        for i, page in enumerate(pages[:10]):
            result = reader.readtext(np.array(page))
            sivun_teksti = " ".join([r[1] for r in result])
            print(f"📄 OCR-sivu {i+1}: {len(sivun_teksti)} merkkiä")
            text_output += sivun_teksti + "\n"

        if text_output.strip():
            kirjaa_ocr_varoitus(file_path, "✅ Teksti luettu EasyOCR:lla (RapidOCR epäonnistui).")
        else:
            kirjaa_ocr_varoitus(file_path, "⚠️ EasyOCR ei löytänyt tekstiä.")

        return text_output.strip()

    except Exception as e:
        kirjaa_ocr_varoitus(file_path, f"EasyOCR epäonnistui: {e}")
        print(f"⚠️ EasyOCR-virhe: {e}")
        return ""

# ===========================================================
# Docling-esikäsittely (RapidOCR + fallback)
# ===========================================================
def prosessoi_doclingilla(file_path: str):
    """Käsittelee dokumentin Doclingilla ja tarvittaessa EasyOCR:lla."""
    raw_path = Path(file_path)
    output_file = PROCESSED_DIR / f"{raw_path.stem}_clean.json"

    if not raw_path.exists():
        kirjaa_ocr_varoitus(file_path, "❌ Tiedostoa ei löydy.")
        return []

    try:
        if output_file.exists():
            print(f"📂 Käytetään välimuistissa olevaa tiedostoa: {output_file}")
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            print(f"🧠 Prosessoidaan dokumentti Doclingilla: {file_path}")
            converter = DocumentConverter()
            result = converter.convert(file_path)
            text_output = result.document.export_to_markdown()

            if FORCE_EASYOCR or not text_output.strip():
                kirjaa_ocr_varoitus(file_path, "Docling ei löytänyt tekstiä – kokeillaan EasyOCR:ia.")
                text_output = tee_easyocr_varamenetelma(file_path)
            else:
                kirjaa_ocr_varoitus(file_path, "✅ Teksti luettu Docling/RapidOCR:lla.")

            if not text_output.strip():
                kirjaa_ocr_varoitus(file_path, "⚠️ OCR ei onnistunut kummallakaan menetelmällä.")
                return []

            data = {"text": text_output}
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"💾 Puhdistettu dokumentti tallennettu: {output_file}")

        text = data.get("text", "")
        words = text.split()
        chunk_size = 500
        text_blocks = [
            " ".join(words[i:i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]

        print(f"✅ Docling-käsittelystä saatiin {len(text_blocks)} tekstikappaletta.")
        return text_blocks

    except Exception as e:
        kirjaa_ocr_varoitus(file_path, f"OCR- tai Docling-virhe: {e}")
        print(f"⚠️ Ohitetaan {file_path}: {e}")
        return []

# ===========================================================
# FAISS-indeksi
# ===========================================================
def rakenna_faiss_indeksi(base_docs_dir=None, index_path=None):
    """Rakentaa FAISS-indeksin kaikista dokumenteista."""
    if base_docs_dir is None:
        base_docs_dir = BASE_DIR / "docs/originals"
    if index_path is None:
        index_path = BASE_DIR / "docs/indexes/combined_index.faiss"

    docs_path = Path(base_docs_dir)
    index_file = Path(index_path)
    meta_file = index_file.with_suffix(".meta.json")

    docs_path.mkdir(parents=True, exist_ok=True)
    index_file.parent.mkdir(parents=True, exist_ok=True)

    # Käytetään olemassa olevaa indeksiä, jos mikään ei ole muuttunut
    if index_file.exists() and meta_file.exists():
        originals = list(docs_path.glob("*"))
        index_mtime = index_file.stat().st_mtime
        changed = [f.name for f in originals if f.stat().st_mtime > index_mtime]
        if not changed:
            print(f"📂 Käytetään olemassa olevaa FAISS-indeksiä: {index_path}")
            index = faiss.read_index(str(index_file))
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)["metadata"]

            passages = []
            for pfile in PROCESSED_DIR.glob("*_clean.json"):  # ✅ korjattu
                with open(pfile, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    text = data.get("text", "")
                    words = text.split()
                    chunks = [" ".join(words[i:i + 300]) for i in range(0, len(words), 300)]
                    passages.extend(chunks)
            print(f"✅ Latauksessa {len(passages)} kappaletta välimuistista.")
            return index, passages, metadata

    supported_exts = [".pdf", ".docx", ".txt"]
    files = [f for f in docs_path.iterdir() if f.suffix.lower() in supported_exts]
    if not files:
        raise FileNotFoundError(f"❌ Ei tuettuja dokumentteja hakemistossa {docs_path}")

    print(f"📄 Löytyi {len(files)} dokumenttia käsiteltäväksi.\n")

    all_passages, metadata = [], []
    for f in tqdm(files, desc="📘 Käsitellään dokumentteja", unit="tiedosto"):
        passages = prosessoi_doclingilla(str(f))
        all_passages.extend(passages)
        metadata.extend([(f.name, i) for i in range(len(passages))])
        print(f"\n📄 Valmis: {f.name} → {len(passages)} tekstikappaletta.\n" + "-" * 60)

    print(f"\n✅ Kaikki dokumentit käsitelty. Yhteensä {len(all_passages)} tekstikappaletta.\n")

    ctx_model = "facebook/dpr-ctx_encoder-single-nq-base"
    ctx_tokenizer = AutoTokenizer.from_pretrained(ctx_model, use_fast=True)
    ctx_encoder = DPRContextEncoder.from_pretrained(ctx_model)
    ctx_encoder.eval()

    embeddings = []
    for i in range(0, len(all_passages), 4):
        batch = all_passages[i:i + 4]
        inputs = ctx_tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = ctx_encoder(**inputs).pooler_output
        embeddings.extend(outputs.cpu().numpy())

    embeddings = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(index_file))

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata}, f, ensure_ascii=False, indent=2)

    print(f"💾 FAISS-indeksi tallennettu: {index_file}")
    print(f"💾 Metatiedot tallennettu: {meta_file}")

    return index, all_passages, metadata

# ===========================================================
# Haku ja vastaus
# ===========================================================
def hae_kappaleet(kysymys: str, index, passages: list[str], k: int = 3):
    print(f"🔎 Haetaan {k} parasta kappaletta kysymykseen: {kysymys}")
    Q_MODEL = "facebook/dpr-question_encoder-single-nq-base"
    q_tokenizer = AutoTokenizer.from_pretrained(Q_MODEL, use_fast=True)
    q_encoder = DPRQuestionEncoder.from_pretrained(Q_MODEL)
    q_encoder.eval()

    q_inputs = q_tokenizer(kysymys, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        q_emb = q_encoder(**q_inputs).pooler_output.cpu().numpy()

    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, k)
    max_valid = len(passages)
    idxs = np.clip(idxs, 0, max_valid - 1)
    haetut = [passages[i] for i in idxs[0] if 0 <= i < len(passages)]

    if not haetut:
        print("⚠️ Ei haettuja kappaleita. Tarkista OCR tai FAISS-indeksi.")
    return haetut

def generoi_vastaus(kysymys: str, konteksti: list[str]):
    print("\n⚙️ Generoidaan vastaus mallilla Alpacazord-Viking-7B...")
    model_name = "mpasila/Alpacazord-Viking-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()

    def muodosta_prompt(konteksti, kysymys, max_ctx_tokens=2000):
        sysmsg = (
            "Toimit asiantuntevana tekoälyavustajana, joka perustaa vastauksensa yksinomaan annettuun kontekstiin. "
            "Älä keksi omia tietoja tai arvaile. Jos kontekstissa ei ole vastausta, sano: "
            "'En voi varmuudella vastata tähän annettujen tietojen perusteella.'\n\n"
            "Kirjoita vastaus selkeällä suomen kielellä. Käytä tarvittaessa numeroituja tai lueteltuja kohtia, "
            "ja korosta keskeiset käsitteet **lihavoimalla**. Älä toista kysymystä. "
            "Pidä vastaus ytimekkäänä ja perusteltuna vain tekstin sisältöön tukeutuen.\n\n"
        )

        header = f"Kysymys: {kysymys}\n\nKonteksti:\n"
        ctx = ""
        for i, p in enumerate(konteksti):
            ehdokas = ctx + f"[Kappale {i+1}]\n{p}\n\n"
            if len(tokenizer.encode(sysmsg + header + ehdokas)) > max_ctx_tokens:
                break
            ctx = ehdokas
        return f"{sysmsg}{header}{ctx}Vastaus:"

    prompt = muodosta_prompt(konteksti, kysymys)

    # Encode the prompt safely
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1800
    ).to(model.device)

    # Generation settings: tuned for factual, grounded answers
    generation_config = {
        "max_new_tokens": 700,          # enough for a long answer
        "temperature": 0.35,            # lower = more deterministic
        "top_p": 0.85,                  # keep some creativity
        "do_sample": True,              # still sample for natural phrasing
        "repetition_penalty": 1.3,      # stronger penalty = less repetition
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "no_repeat_ngram_size": 3,      # prevent repetitive phrases
    }

    # Generate the answer
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_config)

    # Decode only the new tokens (after the prompt)
    answer = tokenizer.decode(
        output_ids[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()

    # Add a short sanity check for very short or empty answers
    if len(answer) < 20:
        answer += "\n(Huom: vastaus jäi lyhyeksi tai konteksti ei sisältänyt riittävästi tietoa.)"

    return answer


# ===========================================================
# Pääohjelma
# ===========================================================
def main(kysymys_override=None):
    print("🚀 Käynnistetään RAG-putki...\n")

    # 1️⃣ Rakennetaan tai ladataan FAISS-indeksi
    index, passages, metadata = rakenna_faiss_indeksi()

    # 2️⃣ Kysymys – joko komentoriviltä tai oletus
    kysymys = kysymys_override or "Millä tavalla merkkaat lähteen lähdeluetteloon, jos käytät verkkosivua?"
    print(f"\n🧭 Kysymys: {kysymys}\n")

    # 3️⃣ Haetaan useampi konteksti (5–7 on hyvä kompromissi)
    kappaleet = hae_kappaleet(kysymys, index, passages, k=5)

    if not kappaleet:
        print("⚠️ Ei kappaleita analysoitavaksi.")
        return

    # 4️⃣ Tulostetaan kontekstit (diagnostiikka)
    print("\n📚 HAKUTULOKSET:")
    for i, kappale in enumerate(kappaleet, 1):
        preview = kappale[:500].replace("\n", " ")
        print(f"\n[Konteksti {i}] {preview}...")
        print("-" * 60)
    
    # Tulosta paras FAISS-score
    scores, idxs = index.search(
    DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    (**AutoTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base", use_fast=True)(
        kysymys, return_tensors="pt", truncation=True, max_length=512
    )).pooler_output.detach().cpu().numpy(), 5)


    print(f"\n📈 Paras FAISS-osuma: {scores[0][0]:.3f}")
    if scores[0][0] < 0.5:
        print("⚠️ Heikko osuma – dokumentti ei ehkä sisällä vastausta.")

    # 5️⃣ Generoidaan vastaus
    vastaus = generoi_vastaus(kysymys, kappaleet)

    # 6️⃣ Tulostetaan lopullinen vastaus
    print("\n" + "=" * 70)
    print("🎯 LOPULLINEN VASTAUS (Viking-7B)")
    print("=" * 70)
    print(f"\nKysymys:\n{kysymys}\n")
    print(f"Vastaus:\n{vastaus}\n")
    print("=" * 70)


def siivoa_muisti():
    torch.cuda.empty_cache()
    gc.collect()
    print("🧹 GPU- ja muistiresurssit vapautettu.")
if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            kysymys = " ".join(sys.argv[1:])
            main(kysymys)
        else:
            main()
    finally:
        siivoa_muisti()
        
