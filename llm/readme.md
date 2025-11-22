---

# 📘 LLM-RAG ja QLoRA -järjestelmän README

## 📂 Projektin yleiskuvaus

Tämä projekti sisältää kaksi toisistaan selkeästi eriytettyä kokonaisuutta:

1. **RAG-järjestelmä (Retrieval-Augmented Generation)**
   – Dokumenttien esikäsittely, OCR, jäsentely, embeddingit, Faiss-haku ja Viking-7B-mallilla generointi.

2. **QLoRA-hienosäätöympäristö**
   – Kevyt LoRA-adapterikoulutus Viking-7B-mallille 4-bit kvantisoinnilla (NF4).

Eri ympäristöt on erotettu toisistaan riippuvuuskonfliktien estämiseksi ja ylläpidettävyyden varmistamiseksi.

---

# 📁 Kansiorakenne

```
llm/
├── src/                      # RAG-pipeline
│   ├── main.py               # pipeline-kokonaisajo
│   ├── ocr_utils.py          # OCR (PaddleOCR / Tesseract)
│   ├── parsing_utils.py      # PDF ja DOCX jäsentely (Unstructured)
│   ├── indexing.py           # Embedding + Faiss-indeksit
│   ├── retrieval.py          # Haku (SBERT, Faiss)
│   └── generation.py         # Viking-7B + strict generation
│
├── venv/                     # RAG-ympäristön virtuaaliympäristö
│
├── docs/                     # Dokumenttien käsittelyyn liittyvät hakemistot
│   ├── originals/
│   ├── processed/
│   └── indexes/
│
├── logs/                     # Lokitiedostot
│
├── pipeline/                 # QLoRA-koulutusympäristö
│   ├── data/                 # train.json ja val.json
│   ├── output/               # LoRA-adapterien tallennus
│   ├── venv/                 # QLoRA-virtuaaliympäristö
│   ├── config_qlora_viking7b.json
│   ├── train_qlora.py
│   ├── infer_qlora.py
│   └── requirements-pipeline.txt
│
└── setup.sh                  # RAG-ympäristön asennusskripti
│
└── requirements.txt          # RAG-ympäristön riippuvuudet
```

---

# 🚀 RAG-pipeline

RAG-järjestelmä koostuu seuraavista vaiheista:

1. **Dokumenttien esikäsittely**

   * Unstructured (0.11.6)
   * PDFMiner, PyMuPDF
   * PaddleOCR (fallback)

2. **Jäsentely ja pilkkominen**

   * metadata, otsikkotasot, sivunumerot
   * chunking → tallennus `docs/processed/`

3. **Embedding-laskenta**

   * Sentence-transformers: *TurkuNLP/sbert-cased-finnish-paraphrase*
   * normalisoidut vektorit

4. **Faiss-indeksi**

   * FlatIP (dot-product)
   * tallennus `docs/indexes/`

5. **Strict Retrieval**

   * semanttinen threshold-filtteri
   * vain aidosti relevantit kappaleet kelpaavat

6. **Strict Generation**

   * Viking-7B (mpasila/Alpacazord-Viking-7B)
   * Generointi sallitaan vain, jos:

     * konteksti löytyy Faissista
     * semanttinen match ≥ threshold
   * muuten: *"En löydä varmaa ohjetta annetuista lähteistä."*

---

# 🔧 RAG-ympäristön asennus

### 1. Aja setup.sh

```
bash setup.sh
```

Tämä:

* asentaa APT-riippuvuudet
* luo virtuaaliympäristön `llm/venv`
* asentaa toimivan RAG-requirements-tiedoston
* asentaa PaddleOCR:n
* tarkistaa tärkeimmät paketit

### 2. Aktivoi venv

```
source llm/venv/bin/activate
```

### 3. Aja koko pipeline

```
python llm/src/main.py
```

---

# 🧪 QLoRA-koulutusympäristö

QLoRA-koulutus on täysin oma erillinen kokonaisuus, jotta se ei riko RAG-ympäristöä.

### 1. Siirry pipeline-kansioon

```
cd llm/pipeline
```

### 2. Luo erillinen virtualenv

```
python3 -m venv venv
source venv/bin/activate
```

### 3. Asenna riippuvuudet

```
pip install -r requirements-pipeline.txt
```

Vain QLoRAa varten:

* transformers
* peft
* bitsandbytes
* datasets
* accelerate

### 4. Koulutus

```
python train_qlora.py
```

### 5. Testaus

```
python infer_qlora.py
```

LoRA-adapterit tallentuvat:

```
pipeline/output/<mallin_nimi>/
```

---

# 📦 JSON-koulutusdata

QLoRA käyttää täsmälleen seuraavaa formaattia:

```json
{
  "instruction": "Opiskelijan kysymys",
  "input": "",
  "output": "Savonian ohjeiden mukainen vastaus"
}
```

---

# 📊 Projektin kaksi ympäristöä

Tämä projekti käyttää kahta täysin erillistä ympäristöä:

| Tarkoitus                 | Polku           | venv                | Sisältö                             |
| ------------------------- | --------------- | ------------------- | ----------------------------------- |
| **RAG-järjestelmä**       | `llm/`          | `llm/venv`          | OCR, Unstructured, Faiss, Viking-7B |
| **QLoRA-mallin koulutus** | `llm/pipeline/` | `llm/pipeline/venv` | transformers, peft, bitsandbytes    |

Tämä ratkaisu estää kirjastokonfliktit ja mahdollistaa tuotantovalmiin arkkitehtuurin.

---

# 📌 Yhteenveto

* RAG-pipeline toimii vakaasti Torch 2.1.2 + CUDA 12.1 -ympäristössä.
* Unstructured 0.11.6 ja PaddleOCR on lukittu toimiviin versioihin.
* Strict Retrieval + Strict Generation takaavat faktapohjaisuuden.
* QLoRA on eriytetty omaan ympäristöön ja valmis jatkokoulutukseen.

---
