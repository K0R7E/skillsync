# backend/rag_engine.py
import os
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from backend.database import load_vectorstore

# LLM konfiguráció
llm = OllamaLLM(model="llama3")

# Prompt Template - Szigorú utasítás az AI-nak
RAG_PROMPT_TEMPLATE = """
Te a SkillSync AI asszisztense vagy. Az alábbi kontextus alapján válaszolj a kérdésre.
Ha nem találod a választ a kontextusban, mondd azt, hogy nem tudod, ne találj ki semmit!

KONTEXTUS:
{context}

KÉRDÉS:
{question}

VÁLASZ (magyarul, lényegretörően):
"""

def get_response(query):
    # 1. Vektoradatbázis betöltése
    vector_db = load_vectorstore()
    if not vector_db:
        return "Hiba: Az adatbázis üres. Tölts fel egy PDF-et előbb!", []

    # 2. Keresés (Hasonlóság alapján a top 3 legrelevánsabb részt kérjük le)
    docs = vector_db.similarity_search(query, k=3)
    
    # 3. Kontextus összefűzése
    context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
    
    # 4. Prompt összeállítása
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    formatted_prompt = prompt.format(context=context_text, question=query)
    
    # 5. LLM hívás
    response_text = llm.invoke(formatted_prompt)
    
    # 6. Források kinyerése metaadatokból
    sources = []
    for doc in docs:
        source_name = os.path.basename(doc.metadata.get('source', 'Ismeretlen'))
        page_num = doc.metadata.get('page', '?') + 1 # 0-ról indul az indexelés
        sources.append(f"{source_name} ({page_num}. oldal)")
    
    # Duplikált források szűrése
    unique_sources = list(set(sources))
    
    return response_text, unique_sources