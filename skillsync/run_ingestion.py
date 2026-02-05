# run_ingestion.py
import os
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Biztosítjuk, hogy a Python látja a backend mappát
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.ingestion import load_and_chunk_pdf
from backend.database import create_or_update_vectorstore

# Cseréld le a fájlnevet a sajátodra a data mappában!
test_pdf = "data/teszt_dokumentum.pdf"

def main():
    if not os.path.exists(test_pdf):
        print(f"❌ Hiba: Nem találom a fájlt: {test_pdf}")
        print("Helyezz egy PDF-et a 'data' mappába ezen a néven.")
        return

    try:
        print(f"🚀 Folyamat indítása: {test_pdf}")
        
        # 4. lépés: Ingestion
        chunks = load_and_chunk_pdf(test_pdf)
        
        # 5. lépés: Vektorizálás és mentés
        create_or_update_vectorstore(chunks)
        
        print("\n✅ SIKER! A PDF-et feldolgoztuk és a vektoradatbázis elkészült.")
        print("Mappa: vectorstore/db_faiss/")
        
    except Exception as e:
        print(f"💥 Hiba történt a futtatás során: {e}")

if __name__ == "__main__":
    main()