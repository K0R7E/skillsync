# backend/database.py
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import os

# Lokális embedding modell konfigurációja
# Fontos: Előtte futtasd: ollama pull nomic-embed-text
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

VECTOR_DB_PATH = "vectorstore/db_faiss"

def create_or_update_vectorstore(chunks):
    """Vektoradatbázis létrehozása vagy frissítése lokálisan."""
    
    if os.path.exists(VECTOR_DB_PATH):
        # Ha már létezik, betöltjük és hozzáadjuk az újat
        vector_db = FAISS.load_local(
            VECTOR_DB_PATH, 
            embedding_model, 
            allow_dangerous_deserialization=True # Lokális környezetben biztonságos
        )
        vector_db.add_documents(chunks)
    else:
        # Új adatbázis létrehozása
        vector_db = FAISS.from_documents(chunks, embedding_model)

    # Mentés a vectorstore/db_faiss mappába
    vector_db.save_local(VECTOR_DB_PATH)
    print(f"💾 Vektoradatbázis elmentve ide: {VECTOR_DB_PATH}")
    return vector_db

def load_vectorstore():
    """Létező adatbázis betöltése kereséshez."""
    if os.path.exists(VECTOR_DB_PATH):
        return FAISS.load_local(
            VECTOR_DB_PATH, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
    return None