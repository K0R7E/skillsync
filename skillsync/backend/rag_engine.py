import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.database import load_vectorstore
from flashrank import Ranker, RerankRequest

# Model és Reranker inicializálása
llm = ChatOllama(model="llama3.2:3b", temperature=0)
# CPU-barát, gyors reranker
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="opt/flashrank")

# --- QUERY EXPANSION PROMPT ---
expansion_prompt = ChatPromptTemplate.from_template("""
Te egy AI asszisztens vagy. Generálj 3 különböző változatot a kérdésből magyarul.
Eredeti kérdés: {question}
Csak a 3 kérdést add meg, számozás nélkül, soronként!
""")
expansion_chain = expansion_prompt | llm | StrOutputParser()

def get_streaming_response(question, history, tenant_id="default"):
    # 1. Kérdés bővítése
    try:
        expanded_text = expansion_chain.invoke({"question": question})
        queries = [question] + [q.strip() for q in expanded_text.strip().split("\n") if q.strip()][:3]
    except:
        queries = [question]

    # 2. Dokumentumok begyűjtése a hibrid keresővel
    retriever = load_vectorstore(tenant_id)
    if not retriever:
        yield "Még nincsenek feltöltött dokumentumok."
        return

    all_docs = []
    for q in queries:
        all_docs.extend(retriever.invoke(q))
    
    unique_docs = {doc.page_content: doc for doc in all_docs}.values()
    
    # 3. RERANKING
    ranker_input = [
        {"id": i, "text": doc.page_content, "meta": doc.metadata} 
        for i, doc in enumerate(unique_docs)
    ]
    
    rerank_request = RerankRequest(query=question, passages=ranker_input)
    results = ranker.rerank(rerank_request)
    
    top_results = results[:3] 
    
    context_parts = []
    sources = []
    
    for res in top_results:
        context_parts.append(res['text'])
        
        # A FlashRank a metaadatokat általában közvetlenül a 'meta' kulcsba teszi,
        # de néha érdemes ellenőrizni, hogy létezik-e
        meta = res.get('meta', {})
        if not meta: # Biztonsági mentőöv, ha a meta üres lenne
            meta = {k: v for k, v in res.items() if k not in ['id', 'text', 'score']}
            
        fname = meta.get('filename', 'Ismeretlen fájl')
        page = meta.get('page', '?')
        
        try:
            # Kezeljük, ha a page None vagy nem szám
            page_num = int(float(page)) + 1 if page != '?' else '?'
        except (ValueError, TypeError):
            page_num = page
            
        sources.append(f"{fname} (oldal: {page_num})")

    context = "\n\n".join(context_parts)
    unique_sources = sorted(list(set(sources)))

    # 4. Válasz generálása
    qa_prompt = ChatPromptTemplate.from_template("""
    Használd az alábbi kontextust. Válaszolj magyarul.
    Kontextus: {context}
    Kérdés: {question}
    """)
    
    final_chain = qa_prompt | llm | StrOutputParser()
    
    for chunk in final_chain.stream({"context": context, "question": question}):
        yield chunk

    # A stream végén küldjük el a forrásokat
    if unique_sources:
        yield f"\n\n📚 **Források:** {', '.join(unique_sources)}"

def get_response(query, history=None, tenant_id="default"):
    """
    Szinkron változat. Összegyűjti a stream minden darabját, 
    beleértve a végére fűzött forrásokat is.
    """
    if history is None: 
        history = []
    
    full_response = ""
    # Végigzongorázzuk a generátort
    for chunk in get_streaming_response(query, history, tenant_id):
        full_response += chunk
    
    # Itt már nem kell külön lista a forrásoknak, 
    # mert a full_response tartalmazza a "📚 Források" részt a végén.
    return full_response