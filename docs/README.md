# 🧠 Generatiivinen tekoäly opinnäytetyöprosessin tukena Savonialla

### 📍 Savonia AMK & DigiCenter - yhteistyöprojekti

Tämä projekti kehittää **paikallisesti ajettavan kielimallipohjaisen sovelluksen**, joka tukee opiskelijaa opinnäytetyöprosessin eri vaiheissa.  
Ratkaisussa hyödynnetään **Viking 7B -kielimallia**, joka toimii **NVIDIA Triton Inference Serverillä** DGX A100 -palvelimella.  
Kielimallia laajennetaan **RAG-tekniikalla** (Retrieval-Augmented Generation), joka hyödyntää Savonian omia ohjeita ja dokumentteja.

---

## 🎯 Tavoite

Rakentaa **toimiva prototyyppi**, jota Savonian opiskelijat voivat testata suljetussa ympäristössä.  
Prototyyppi yhdistää:

- paikallisesti ajetun kielimallin (Viking 7B)  
- FAISS-indeksin Savonian ohjeaineistolle  
- FastAPI-backendin ja React-frontendin  

---

## 🧩 Tekninen kokonaisuus

| Osa | Teknologia | Kuvaus |
|------|-------------|--------|
| **Inferenssi** | NVIDIA Triton Server | Viking 7B -mallin ajo DGX A100:lla |
| **Backend** | FastAPI (Python) | Kysymysten käsittely, RAG, FAISS-haku |
| **Frontend** | React (Vite, Tailwind) | Chat-käyttöliittymä opiskelijalle |
| **Tietokanta** | SQLite / PostgreSQL | Palautteiden tallennus |
| **Versionhallinta** | Azure DevOps (Git) | Koodi, Dockerfilet, dokumentaatio |
| **Kontitus** | Docker + docker-compose | Jokainen osa erillisenä konttina |

---

## 🗂️ Hakemistorakenne

```bash
savoniallm/
├── backend/
│   ├── app/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   └── Dockerfile
├── triton/
│   └── model_repository/
│       └── viking7b/
│           ├── config.pbtxt
│           └── model.plan
├── vector_store/
│   └── faiss_index.bin
├── docker-compose.yml
├── .env.template
└── docs/
    └── README.md
