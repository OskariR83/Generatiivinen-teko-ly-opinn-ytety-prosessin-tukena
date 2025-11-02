"""
RAG-putki: Docling + DPR + FAISS (CPU) + Viking-7B / Alpacazord-Viking-7B / Viking-13B-GGUF
Projekti: GENERATIIVINEN TEKOÄLY OPINNÄYTETYÖPROSESSIN TUKENA

Kuvaus:
    Tämä ohjelma muodostaa päästä päähän RAG-prosessin, joka:
    - esikäsittelee ja muuntaa dokumentit tekstimuotoon Docling-kirjastolla
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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DPRContextEncoder,
    DPRQuestionEncoder,
)
from docling.document_converter import DocumentConverter
from transformers.utils import logging
from datetime import datetime
import gc

# Vähennetään Hugging Face -kirjaston lokitusta
logging.set_verbosity_error()

# ===========================================================
# ✅ Polkujen ja kansioiden määritys
# ===========================================================
BASE_DIR = Path(__file__).resolve().parents[2]
DOCS_DIR = BASE_DIR / "docs"
ORIGINALS_DIR = DOCS_DIR / "originals"
PROCESSED_DIR = DOCS_DIR / "processed"
INDEX_DIR = DOCS_DIR / "indexes"

for d in [DOCS_DIR, ORIGINALS_DIR, PROCESSED_DIR, INDEX_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"📂 Projektihakemisto: {BASE_DIR}")
print(f"📄 Alkuperäiset dokumentit: {ORIGINALS_DIR}")
print(f"🧹 Prosessoidut tiedostot: {PROCESSED_DIR}")
print(f"🧠 Indeksit: {INDEX_DIR}")


# ===========================================================
# Vaihe 1: Docling-esikäsittely ja OCR-tarkistus
# ===========================================================
def kirjaa_ocr_varoitus(file_path, viesti):
    """
    Kirjaa OCR-varoitukset tiedostoon logs/ocr_failures.log.
    Tallentaa aikaleiman, tiedoston nimen ja varoitusviestin.
    """
    log_file = Path("logs/ocr_failures.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    aikaleima = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{aikaleima}] {file_path}: {viesti}\n")

    print(f"⚠️ OCR-varoitus kirjattu: {file_path}")


def prosessoi_doclingilla(file_path: str):
    """
    Käsittelee yksittäisen dokumentin (PDF, DOCX, TXT) Docling-kirjastolla.

    Vaiheet:
    1. Tarkistaa onko aiemmin prosessoitu versio tallennettu (välimuisti)
    2. Jos ei ole, muuntaa dokumentin tekstimuotoon ja tallentaa sen JSON-muodossa
    3. Jakaa tekstin noin 500 sanan osiin (kappaleisiin)
    4. Kirjaa OCR-varoitukset, jos tekstiä ei löydy tai Docling epäonnistuu
    """
    raw_path = Path(file_path)
    output_file = PROCESSED_DIR / f"{raw_path.stem}_clean.json"

    try:
        # Käytä välimuistia, jos olemassa
        if output_file.exists():
            print(f"📂 Käytetään välimuistissa olevaa Docling-tiedostoa: {output_file}")
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            print(f"🧠 Prosessoidaan dokumentti Doclingilla: {file_path}")
            converter = DocumentConverter()
            result = converter.convert(file_path)
            text_output = result.document.export_to_markdown()

            if not text_output.strip():
                kirjaa_ocr_varoitus(file_path, "Docling ei löytänyt tekstiä – mahdollinen OCR-virhe.")

            data = {"text": text_output}

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"💾 Puhdistettu dokumentti tallennettu: {output_file}")

        # Jaetaan teksti kappaleisiin
        text_blocks = []
        if "sections" in data:
            for section in data["sections"]:
                text = section.get("text", "").strip()
                if text:
                    text_blocks.append(text)
        elif "text" in data:
            text = data["text"]
            if not text.strip():
                kirjaa_ocr_varoitus(file_path, "Docling tuotti tyhjän tekstin.")
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
# Vaihe 2: FAISS-indeksin rakentaminen
# ===========================================================
def rakenna_faiss_indeksi(base_docs_dir="docs/originals", index_path="docs/indexes/combined_index.faiss"):
    """
    Rakentaa FAISS-indeksin kaikista dokumenteista.
    Tarkistaa välimuistin, käsittelee uudet dokumentit ja laskee DPR-upotukset.
    """
    docs_path = Path(base_docs_dir)
    index_file = Path(index_path)
    meta_file = index_file.with_suffix(".meta.json")

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
            for pfile in Path("docs/processed").glob("*_clean.json"):
                with open(pfile, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "text" in data:
                        words = data["text"].split()
                        chunk_size = 300
                        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
                        passages.extend(chunks)
            print(f"✅ Latauksessa {len(passages)} kappaletta välimuistista.")
            return index, passages, metadata
        else:
            print(f"♻️ Rakennetaan indeksi uudelleen – muuttuneet tiedostot: {', '.join(changed)}")

    print(f"🧠 Haetaan dokumentit kansiosta: {docs_path}")
    supported_exts = [".pdf", ".docx", ".txt"]
    files = [f for f in docs_path.iterdir() if f.suffix.lower() in supported_exts]
    if not files:
        raise FileNotFoundError(f"❌ Ei tuettuja dokumentteja hakemistossa {docs_path}")
    print(f"📄 Löytyi {len(files)} dokumenttia käsiteltäväksi.")

    all_passages, metadata = [], []
    for f in files:
        try:
            print(f"🧩 Käsitellään: {f.name} ...")
            passages = prosessoi_doclingilla(str(f))
            all_passages.extend(passages)
            metadata.extend([(f.name, i) for i in range(len(passages))])
        except Exception as e:
            print(f"⚠️ Ohitetaan {f.name}: {e}")

    print(f"✅ Yhteensä {len(all_passages)} kappaletta luotu.")

    ctx_model = "facebook/dpr-ctx_encoder-single-nq-base"
    ctx_tokenizer = AutoTokenizer.from_pretrained(ctx_model, use_fast=True)
    ctx_encoder = DPRContextEncoder.from_pretrained(ctx_model)
    ctx_encoder.eval()

    embeddings_list = []
    for i in range(0, len(all_passages), 4):
        batch = all_passages[i:i + 4]
        inputs = ctx_tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = ctx_encoder(**inputs).pooler_output
        embeddings_list.extend(outputs.cpu().numpy())

    embeddings = np.array(embeddings_list, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(index_file))
    print(f"💾 FAISS-indeksi tallennettu: {index_file}")

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata}, f, ensure_ascii=False, indent=2)
    print(f"💾 Metatiedot tallennettu: {meta_file}")

    return index, all_passages, metadata


# ===========================================================
# Vaihe 3: Kappaleiden haku kysymyksen perusteella
# ===========================================================
def hae_kappaleet(kysymys: str, index, passages: list[str], k: int = 3):
    """
    Hakee kysymystä vastaavat kappaleet FAISS-indeksistä.
    Korjattu versio, joka estää virheelliset indeksiviittaukset.
    """
    print(f"🔎 Haetaan {k} parasta kappaletta kysymykseen: {kysymys}")

    Q_MODEL = "facebook/dpr-question_encoder-single-nq-base"
    from transformers import AutoTokenizer
    q_tokenizer = AutoTokenizer.from_pretrained(Q_MODEL, use_fast=True)
    q_encoder = DPRQuestionEncoder.from_pretrained(Q_MODEL)
    q_encoder.eval()

    q_inputs = q_tokenizer(
        kysymys,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        q_emb = q_encoder(**q_inputs).pooler_output.cpu().numpy()

    faiss.normalize_L2(q_emb)
    scores, idxs = index.search(q_emb, k)

    # ✅ Estetään IndexError virhe, jos FAISS palauttaa liian suuren indeksin
    max_valid = len(passages)
    idxs = np.clip(idxs, 0, max_valid - 1)

    # ✅ Varmistetaan myös ettei tyhjiä indeksejä käytetä
    haetut = []
    for i in idxs[0]:
        if 0 <= i < len(passages):
            haetut.append(passages[i])

    print(f"✅ Haettu {len(haetut)} kappaletta.")
    for j, passage in enumerate(haetut, 1):
        preview = passage[:200] + "..." if len(passage) > 200 else passage
        print(f"\n--- Kappale {j} ---\n{preview}")

    if not haetut:
        print("⚠️ Ei haettuja kappaleita. Tarkista, että FAISS-indeksi ja käsitellyt tekstit vastaavat toisiaan.")
    return haetut


# ===========================================================
# Vaihe 4: Vastauksen generointi Viking-7B-mallilla
# ===========================================================
def generoi_vastaus(kysymys: str, konteksti: list[str]):
    """
    Tuottaa suomenkielisen vastauksen käyttäen Viking-7B (Alpacazord) -mallia.
    """
    print("\n⚙️ Generoidaan vastaus mallilla Alpacazord-Viking-7B...")

    model_name = "mpasila/Alpacazord-Viking-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    def muodosta_prompt(konteksti, kysymys, max_ctx_tokens=2000):
        system_msg = (
            "Olet asiantunteva tekoälyavustaja ja kirjoitat sujuvaa suomen kieltä. "
            "Vastaa vain annetun kontekstin perusteella. "
            "Jos vastaus ei löydy kontekstista, sano: 'En tiedä varmasti tämän perusteella.' "
            "Käytä luettelomerkkejä ja korosta tärkeät käsitteet **lihavoimalla**.\n\n"
        )
        header = f"Kysymys: {kysymys}\n\nKonteksti:\n"
        ctx = ""
        for i, p in enumerate(konteksti):
            ehdokas = ctx + f"[Kappale {i+1}]\n{p}\n\n"
            if len(tokenizer.encode(system_msg + header + ehdokas)) > max_ctx_tokens:
                break
            ctx = ehdokas
        return f"{system_msg}{header}{ctx}Vastaus:"

    prompt = muodosta_prompt(konteksti, kysymys)
    print("\n🧩 PROMPT-esikatselu:\n", prompt[:500], "...\n")

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1600).to(model.device)
    print(f"📊 Promptin pituus: {inputs.input_ids.shape[1]} tokenia")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=700,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_only = output_ids[0][inputs.input_ids.shape[1]:]
    vastaus = tokenizer.decode(gen_only, skip_special_tokens=True).strip()

    if len(vastaus) < 15:
        vastaus += "\n(Huom. vastaus katkesi tai konteksti oli liian pitkä.)"

    print("✅ Vastauksen generointi valmis.")
    return vastaus


# ===========================================================
# Vaihe 5: Pääsuoritus (CLI-tuki)
# ===========================================================
def main(kysymys_override=None):
    print("🚀 Käynnistetään RAG-putki monelle dokumentille...\n")

    docs_dir = BASE_DIR / "docs/originals"
    index_path = BASE_DIR / "docs/indexes/combined_index.faiss"
    index, passages, metadata = rakenna_faiss_indeksi(docs_dir, index_path)

    kysymys = kysymys_override or "Millä tavalla merkkaat lähteen lähdeluetteloon, jos käytät verkkosivua?"
    print(f">>> Käytettävä kysymys: {kysymys}\n")

    parhaat_kappaleet = hae_kappaleet(kysymys, index, passages, k=3)
    print(f">>> Ensimmäisen kappaleen esikatselu:\n{parhaat_kappaleet[0][:300]}...\n")

    vastaus = generoi_vastaus(kysymys, parhaat_kappaleet)

    print("\n" + "=" * 50)
    print("🎯 LOPULLINEN VASTAUS (Viking-7B)")
    print("=" * 50)
    print(f"\nKysymys: {kysymys}")
    print(f"\nVastaus:\n{vastaus}")
    print("\n" + "=" * 50)


def siivoa_muisti():
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            kysymys_arg = " ".join(sys.argv[1:])
            main(kysymys_arg)
        else:
            main()
    finally:
        siivoa_muisti()
        print("🧹 GPU- ja muistiresurssit vapautettu.")
