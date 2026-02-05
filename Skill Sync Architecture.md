# SkillSync – Architektúra

## 🏗️ Áttekintés

SkillSync egy **offline-first, lokális RAG rendszer**, amely Python backenddel és Ollama modellekkel működik.

## 🔧 Fő komponensek

### 1. UI réteg

* Web UI (FastAPI + simple frontend)
* Funkciók:

  * PDF feltöltés
  * Chat felület
  * Forrásmegjelölések megjelenítése

### 2. API / Backend

* FastAPI
* Endpointok:

  * /upload
  * /chat
  * /reindex

### 3. Ingestion Pipeline

* PDF parser
* Chunkolás (token-alapú)
* Metadata: fájlnév, oldalszám
* Embedding generálás

### 4. Vector Store

* FAISS (lokális)
* Cégenként elkülönített index

### 5. RAG Engine

* Query embedding
* Top-k retrieval
* Context összeállítás
* Prompt template

### 6. LLM réteg

* Ollama
* Cserélhető modellek

## 🔐 Biztonsági modell

* 1 instance = 1 cég
* Lokális filesystem
* Nincs cloud dokumentumforgalom

## 🔄 Frissítések

* Internet csak:

  * modell frissítésre
  * app update-re
* Dokumentumok soha nem szinkronizálódnak

## 🧠 Adatfolyam

```
PDF → Chunk → Embedding → Vector DB
User Query → Embedding → Retrieval → LLM → Answer + Sources
```

## 🛠️ Tech Stack

* Python 3.11+
* FastAPI
* Ollama
* FAISS
* LangChain / LlamaIndex (opcionális)
