# backend/ingestion.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(file_path):
    """PDF betöltése és darabolása metaadatokkal."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"A fájl nem található: {file_path}")

    # 1. PDF betöltése
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
    except Exception as e:
        raise Exception(f"Hiba a PDF beolvasásakor: {e}")

    # 2. Szöveg felosztása (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(pages)
    
    print(f"✅ Feldolgozva: {file_path}")
    print(f"📄 Oldalak száma: {len(pages)}")
    print(f"🧩 Chunkok száma: {len(chunks)}")
    
    return chunks