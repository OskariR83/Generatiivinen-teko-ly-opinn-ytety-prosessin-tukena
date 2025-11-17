# Opinnäytetyön apuri – Backend (FastAPI)

Tämä on opinnäytetyöprojektin backend-sovellus, joka toimii FastAPI:n avulla ja toteuttaa RAG-pohjaisen (Retrieval-Augmented Generation) kielimallirajapinnan. Backend vastaa REST API -rajapinnoista, dokumenttien esikäsittelystä, FAISS-vektorihakujärjestelmästä, session-pohjaisesta keskusteluhistoriasta sekä Viking-7B-kielimallin integraatiosta.

Backend toimii sekä paikallisesti että Savonian DGX A100 -palvelimen hiekkalaatikkoympäristössä.

Huom: Mallien painot (Viking-7B) eivät ole repossa. Ne tallennetaan DGX-palvelimen mallikansioon.

---

## Projektin rakenne

```
backend/
│
├── app/
│   ├── api/
│   │   ├── routes.py         ← Koonti /api/ reiteille
│   │   ├── llm_routes.py     ← LLM/RAG-kyselyt + session-historia
│   │   └── docs_routes.py    ← OCR + dokumenttien esiprosessointi + FAISS-status
│   │
│   ├── core/
│   │   └── config.py         ← .env-muuttujat
│   │
│   ├── database/
│   │   └── connection.py     ← PostgreSQL/SQLModel-yhteys
│   │
│   └── models/
│       └── feedback.py       ← Palaute-taulun tietomalli
│
├── main.py                   ← Sovelluksen käynnistyspiste
├── requirements.txt          ← Riippuvuudet
├── .env                      ← Tietokannan asetukset
└── README.md                 
```

---

## Asennusohjeet

### 1. Siirry backend-kansioon

Avaa PowerShell tai terminaali ja siirry projektin backend-hakemistoon:

```bash
cd "GENERATIIVINEN TEKOÄLY OPINNÄYTETYÖPROSESSIN TUKENA/backend"
```

### 2. Luo ja aktivoi virtuaaliympäristö

```
windows
powershell
py -m venv venv
.\venv\Scripts\Activate


linux
python3 -m venv venv
source venv/bin/activate

```

Jos Windows estää skriptin suorittamisen, voit sallia sen tilapäisesti:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate
```

### 3. Asenna riippuvuudet

```bash
pip install -r requirements.txt
```

Jos `requirements.txt` puuttuu, voit asentaa kirjastot käsin:

```bash
pip install fastapi uvicorn sqlmodel python-dotenv psycopg2-binary
```

### 4. Luo .env-tiedosto

Luo projektin juureen (`backend/`) tiedosto nimeltä `.env` ja lisää sinne tietokantayhteys:

```env
DATABASE_URL=postgresql+psycopg2://postgres:root@localhost:5432/llm_db
```

### 5. Luo tietokanta

Asenna PostgreSQL ja varmista, että palvelu on käynnissä.

Luo tyhjä tietokanta nimeltä `llm_db`:

```sql
CREATE DATABASE llm_db;
```

### 6. Käynnistä sovellus

Käynnistä sovellus backend-kansiosta:

```
paikallisesti
bash
uvicorn main:app --reload

hiekkalaatikossa
uvicorn app.main:app --reload --port 8000

```

### 7. Testaa selaimessa

#### Testireitti
```
http://127.0.0.1:8000/api/test
```
Vastaus:
```json
{ "status": "OK" }
```

#### Swagger UI
```
http://127.0.0.1:8000/docs
```
API:n graafinen testausympäristö

#### Root-reitti
```
http://127.0.0.1:8000
```
Näyttää viestin:
```json
"Paikallisesti ajettava kielimalli – backend toimii!"
```

---

## Vinkit

- Käynnistä komennot aina `backend`-kansiosta, jotta importit toimivat oikein
- Aktivoi virtuaaliympäristö aina ennen `uvicorn`-komentoa
- Ensimmäisellä käynnistyksellä `init_db()` luo tietokannan automaattisesti, jos `DATABASE_URL` on asetettu oikein