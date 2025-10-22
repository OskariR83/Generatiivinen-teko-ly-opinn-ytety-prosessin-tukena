# Opinnäytetyön apuri – Backend (FastAPI)

Tämä on opinnäytetyöprojektin backend-sovellus, joka toimii paikallisesti FastAPI:n avulla.
Sovellus tarjoaa REST API -rajapinnan, johon voidaan myöhemmin liittää kielimalli (LLM) ja tietokantatoiminnot.

---

## Projektin rakenne

```
backend/
│
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py          ← API-reitit (testi & palautteet)
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py          ← ympäristömuuttujien lataus
│   ├── database/
│   │   ├── __init__.py
│   │   └── connection.py      ← tietokantayhteys (SQLModel)
│   └── models/
│       ├── __init__.py
│       └── feedback.py        ← palautetaulun malli
│
├── main.py                    ← sovelluksen käynnistyspiste
├── requirements.txt           ← kirjastot
├── .env                       ← ympäristömuuttujat (esim. DATABASE_URL)
└── README.md                  ← tämä tiedosto
```

---

## Asennusohjeet

### 1. Siirry backend-kansioon

Avaa PowerShell tai terminaali ja siirry projektin backend-hakemistoon:

```bash
cd "GENERATIIVINEN TEKOÄLY OPINNÄYTETYÖPROSESSIN TUKENA/backend"
```

### 2. Luo ja aktivoi virtuaaliympäristö

```powershell
py -m venv venv
.\venv\Scripts\Activate
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

```bash
uvicorn main:app --reload
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