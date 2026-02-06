import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# LangChain importok a feldolgozáshoz
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage

# Saját modulok
from backend.database import create_or_update_vectorstore
from backend.rag_engine import get_streaming_response 

app = FastAPI(title="SkillSync MVP API")

# Statikus fájlok kiszolgálása (ha van CSS/JS az ui mappában)
# app.mount("/static", StaticFiles(directory="ui"), name="static")

# --- ADATSTRUKTÚRÁK ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatQuery(BaseModel):
    message: str
    history: List[ChatMessage] = []
    tenant_id: str = "default" # Hozzáadva az izolációhoz

# --- VÉGPONTOK ---

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "SkillSync"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), tenant_id: str = Form("default")):
    try:
        # 1. Tenant-specifikus tárolás
        upload_dir = f"storage/{tenant_id}"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Betöltés és feldolgozás (itt helyben, vagy az ingestion.py-ból)
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        
        # 3. Vektoradatbázis frissítése a Hybrid keresőhöz
        create_or_update_vectorstore(splits, tenant_id)
        
        return {"message": f"Sikeres feltöltés! Tenant: {tenant_id}, Fájl: {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(query: ChatQuery):
    # History konvertálása LangChain formátumra
    langchain_history = []
    for msg in query.history:
        if msg.role == "human":
            langchain_history.append(HumanMessage(content=msg.content))
        else:
            langchain_history.append(AIMessage(content=msg.content))

    def generate():
        # Átadjuk a query-t, a history-t és a tenant_id-t a hibrid motornak
        for chunk in get_streaming_response(query.message, langchain_history, query.tenant_id):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")
    
@app.get("/")
async def read_index():
    if os.path.exists('ui/index.html'):
        return FileResponse('ui/index.html')
    return {"error": "UI index.html nem található"}