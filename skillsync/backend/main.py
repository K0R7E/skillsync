# backend/main.py
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Saját moduljaink importálása
from backend.ingestion import load_and_chunk_pdf
from backend.database import create_or_update_vectorstore
from backend.rag_engine import get_response

app = FastAPI(title="SkillSync MVP API")

# Adatstruktúra a chat kéréshez
class ChatQuery(BaseModel):
    message: str

# 1. Health Check - Hogy lássuk, fut-e a szerver
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "SkillSync"}

# 2. PDF Feltöltés végpont
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Csak PDF fájlok támogatottak!")

    file_path = f"data/{file.filename}"
    
    # Fájl mentése lokálisan
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Feldolgozás (4-5. lépés újrafelhasználása)
        chunks = load_and_chunk_pdf(file_path)
        create_or_update_vectorstore(chunks)
        return {"message": f"Sikeresen feldolgozva: {file.filename}", "chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 3. Chat végpont
@app.post("/chat")
async def chat(query: ChatQuery):
    try:
        answer, sources = get_response(query.message)
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# UI kiszolgálása
@app.get("/")
async def read_index():
    return FileResponse('ui/index.html')