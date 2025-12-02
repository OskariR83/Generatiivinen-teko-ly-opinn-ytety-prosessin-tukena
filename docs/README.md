# 🧠 Generatiivinen tekoäly opinnäytetyöprosessin tukena Savonialla

### 📍 Savonia AMK & DigiCenter - yhteistyöprojekti

Tämä projekti kehittää **paikallisesti ajettavan kielimallipohjaisen sovelluksen**, joka tukee opiskelijaa opinnäytetyöprosessin eri vaiheissa.  
Ratkaisussa hyödynnetään **Viking 7B -kielimallia**, joka toimii DGX A100 -palvelimella.  
Kielimallia laajennetaan **RAG-tekniikalla** (Retrieval-Augmented Generation), joka hyödyntää Savonian omia ohjeita ja dokumentteja. Ohjeet eivät ole mukana repossa. 

---

## 🎯 Tavoite

Rakentaa **toimiva prototyyppi**, jota Savonian opiskelijat voivat testata suljetussa ympäristössä.  
Prototyyppi yhdistää:

- paikallisesti ajetun kielimallin (Alpacazord/Viking 7B)  
- FAISS-indeksin Savonian ohjeaineistolle  
- FastAPI-backendin ja React-frontendin  

---

## 🧩 Tekninen kokonaisuus


| Osa | Teknologia | Kuvaus |
|------|-------------|--------|
| **LLM** | Kielimallin skriptit | Viking 7B -mallin ajo DGX A100:lla |
| **Backend** | FastAPI (Python) | Kysymysten käsittely, RAG, FAISS-haku |
| **Frontend** | React (Vite, Tailwind) | Chat-käyttöliittymä opiskelijalle |
| **Tietokanta** | PostgreSQL | Palautteiden tallennus |
| **Versionhallinta** | Koodi, Dockerfilet, dokumentaatio |

---
