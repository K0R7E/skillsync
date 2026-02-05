# SkillSync 🧠📄

**SkillSync** egy lokálisan futtatható, biztonságos AI chatbot, amely a céged belső PDF dokumentumaiból segít gyors válaszokat adni – **adatkiszivárgás nélkül**.

## ✨ Miért SkillSync?

* 🔒 100% lokális adatkezelés
* 🧠 RAG-alapú tudáskeresés
* 📄 PDF-fókuszú
* ⚙️ Python + Ollama
* 🚀 Gyors telepítés

## 🧩 Hogyan működik?

1. Feltöltöd a PDF-eket
2. SkillSync indexeli őket lokálisan
3. Kérdezel természetes nyelven
4. Választ kapsz forrásmegjelöléssel

## 🏗️ Architektúra röviden

* Web UI + FastAPI backend
* FAISS vektoradatbázis
* Ollama LLM
* Offline-first működés

## 🔐 Biztonság

* Dokumentumok **nem hagyják el a gépet**
* Cégenként elkülönített instance
* Nincs SaaS, nincs központi adatgyűjtés

## 🚀 Gyors indítás

```bash
pip install skillsync
skillsync start
```

## 🎯 Use case-ek

* Belső dokumentáció keresése
* Onboarding
* IT / HR / jogi tudásbázis

## 🛣️ Roadmap

* Jogosultságkezelés
* Verziózás
* Multi-nyelv
* Fine-tuning

## 🤝 Hozzájárulás

PR-ek és ötletek welcome 🙌

## 📜 Licenc

MIT
