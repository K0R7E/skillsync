import os
from langchain_ollama import ChatOllama # Ezt javítottuk!
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.database import load_vectorstore

# 1. Alapmodell a válaszadáshoz és a kérdésbővítéshez
# Llama 3.2 3B tökéletes erre a feladatra
llm = ChatOllama(model="llama3.2:3b", temperature=0)

# --- QUERY EXPANSION RÉSZ ---
expansion_prompt = ChatPromptTemplate.from_template("""
Te egy AI asszisztens vagy. A feladatod, hogy a felhasználó kérdéséből 3 különböző 
változatot generálj magyar nyelven, hogy segíts a dokumentumok közötti keresésben.
Cél: Különböző megfogalmazásokkal több releváns találatot kapjunk.

Eredeti kérdés: {question}

Csak a 3 új kérdést add meg, soronként, számozás nélkül!
""")

expansion_chain = expansion_prompt | llm | StrOutputParser()

def get_streaming_response(question, history, tenant_id="default"):
    # 2. Kérdés bővítése (Query Expansion)
    try:
        expanded_text = expansion_chain.invoke({"question": question})
        # Tisztítjuk és listába szedjük a kérdéseket
        queries = [question] + [q.strip() for q in expanded_text.strip().split("\n") if q.strip()][:3]
    except Exception as e:
        print(f"DEBUG Expansion hiba: {e}")
        queries = [question]
    
    # print(f"DEBUG: Bővített kérdések: {queries}")

    # 3. Hybrid Retriever betöltése
    retriever = load_vectorstore(tenant_id)
    if not retriever:
        yield "Sajnos még nincs feltöltött dokumentumod ehhez a fiókhoz."
        return

    # 4. Keresés az összes generált kérdéssel
    all_docs = []
    for q in queries:
        # Meghívjuk a hybrid retrieverünket (BM25 + FAISS)
        all_docs.extend(retriever.invoke(q))
    
    # Duplikátumok kiszűrése (page_content alapján)
    unique_docs = {doc.page_content: doc for doc in all_docs}.values()
    context = "\n\n".join([doc.page_content for doc in unique_docs])

    # 5. Végső válasz generálása
    qa_prompt = ChatPromptTemplate.from_template("""
    Te egy professzionális asszisztens vagy. Az alábbi kontextus alapján válaszolj a kérdésre.
    Ha a kontextusban nincs benne a válasz, mondd azt, hogy nem találtam erről információt.
    
    Kontextus: {context}
    Kérdés: {question}
    """)
    
    final_chain = qa_prompt | llm | StrOutputParser()
    
    for chunk in final_chain.stream({"context": context, "question": question}):
        yield chunk

def get_response(query, history=None, tenant_id="default"):
    """Nem streaming változat a kompatibilitás kedvéért."""
    if history is None: history = []
    full_response = ""
    for chunk in get_streaming_response(query, history, tenant_id):
        full_response += chunk
    return full_response, ["Helyi PDF dokumentum"]