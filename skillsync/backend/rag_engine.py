import os
from langchain_ollama import ChatOllama # Ezt javítottuk!
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.database import load_vectorstore
from flashrank import Ranker, RerankRequest

# 1. Alapmodell a válaszadáshoz és a kérdésbővítéshez
# Llama 3.2 3B tökéletes erre a feladatra
llm = ChatOllama(model="llama3.2:3b", temperature=0)
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="opt/flashrank")

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
        yield "Még nincsenek dokumentumok."
        return

    # 2. Dokumentumok begyűjtése (több kérdés alapján)
    all_docs = []
    for q in queries:
        all_docs.extend(retriever.invoke(q))
    
    # Duplikátumok szűrése
    unique_docs = {doc.page_content: doc for doc in all_docs}.values()
    
    # 3. RERANKING LÉPÉS
    # Átalakítjuk a formátumot a FlashRank számára
    ranker_input = [
        {"id": i, "text": doc.page_content, "meta": doc.metadata} 
        for i, doc in enumerate(unique_docs)
    ]
    
    rerank_request = RerankRequest(query=question, passages=ranker_input)
    results = ranker.rerank(rerank_request)
    
    # Csak a legjobb 3 találatot tartjuk meg
    top_results = results[:3]
    context = "\n\n".join([r['text'] for r in top_results])
    
    # print(f"DEBUG: Reranker utáni legjobb pontszám: {top_results[0]['score'] if top_results else 'N/A'}")

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