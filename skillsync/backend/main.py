import os
import shutil
import uuid
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
# LangChain importok
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

# Saját modulok
from backend.database import create_or_update_vectorstore
from backend.database_sql import init_sql_db, save_message_to_db
from backend.rag_engine import get_streaming_response

app = FastAPI(title="SkillSync MVP API")

# --- ADATSTRUKTÚRÁK ---


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []
    tenant_id: str = "default"
    session_id: Optional[str] = None


# --- INICIALIZÁCIÓ ---
@app.on_event("startup")
async def startup_event():
    init_sql_db()


# --- VÉGPONTOK ---


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), tenant_id: str = Form("default")):
    try:
        upload_dir = f"storage/{tenant_id}"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        splits = text_splitter.split_documents(docs)

        create_or_update_vectorstore(splits, tenant_id)

        return {
            "message": f"Sikeres feltöltés! Tenant: {tenant_id}, Fájl: {file.filename}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_endpoint(request: ChatRequest):
    # Session kezelés: ha nincs, generálunk
    session_id = request.session_id or str(uuid.uuid4())

    # 1. Felhasználói üzenet mentése az AUDIT logba
    save_message_to_db(session_id, request.tenant_id, "user", request.message)

    async def event_generator():
        full_ai_response = ""
        # RAG motor hívása
        for chunk in get_streaming_response(
            request.message, request.history, request.tenant_id
        ):
            full_ai_response += chunk
            yield chunk

        # 2. AI válasz mentése a végén (History + Audit)
        save_message_to_db(session_id, request.tenant_id, "assistant", full_ai_response)

    return StreamingResponse(event_generator(), media_type="text/plain")


@app.get("/files/{tenant_id}")
async def list_files(tenant_id: str = "default"):
    upload_dir = f"storage/{tenant_id}"
    if not os.path.exists(upload_dir):
        return []

    file_list = []
    for f in os.listdir(upload_dir):
        if f.endswith(".pdf"):
            file_path = os.path.join(upload_dir, f)
            file_list.append({"name": f, "size": os.path.getsize(file_path)})
    return file_list


@app.delete("/files/{tenant_id}/{filename}")
async def delete_file(tenant_id: str, filename: str):
    file_path = f"storage/{tenant_id}/{filename}"
    if os.path.exists(file_path):
        os.remove(file_path)
        from backend.database import rebuild_index_for_tenant

        rebuild_index_for_tenant(tenant_id)
        return {"message": "Törölve"}
    raise HTTPException(status_code=404, detail="Nem található")


@app.get("/")
async def read_index():
    return FileResponse("ui/index.html")
