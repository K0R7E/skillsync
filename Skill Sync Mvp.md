# SkillSync – MVP Definíció

## 🎯 Cél

Egy **lokálisan futtatható, biztonságos RAG-alapú chatbot**, amely cégen belüli PDF dokumentumokból segíti a tudásmegosztást.

## 🧩 MVP Scope (mi fér bele)

### Kötelező funkciók

* Lokális futtatás Python + Ollama alapon
* PDF feltöltés **csak lokálisan**
* Cégenként elkülönített adat (1 instance = 1 cég)
* RAG pipeline (chunkolás → embedding → vektor DB → válasz)
* Forrásmegjelölés válaszoknál (PDF + oldalszám)
* Egyszerű web UI (upload + chat)
* CLI indítás / konfiguráció

### Biztonság

* Dokumentumok nem hagyják el a gépet
* Lokális vektoradatbázis
* Alap titkosítás (filesystem szint)

### Modellek

* Ollama LLM (pl. Llama / Mistral)
* Lokális embedding modell

## ❌ Nem része az MVP-nek

* Multi-tenant SaaS
* Felhasználói analitika
* Finomhangolás / LoRA
* Külső integrációk (Slack, Teams)
* Cloud sync dokumentumokra

## 🏁 MVP siker kritériumok

* 10–50 PDF stabil kezelése
* Releváns válaszok forrásmegjelöléssel
* Egyszerű telepítés (<10 perc)

## 🔜 Következő lépések (post-MVP)

* Jogosultságkezelés
* Verziózott dokumentumkezelés
* Több nyelv
