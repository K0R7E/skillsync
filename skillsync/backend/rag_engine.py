import os
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from backend.database import load_vectorstore

# LLM beállítása
llm = OllamaLLM(model="llama3")

def format_docs(docs):
    """Dokumentumok összefűzése kontextussá."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_streaming_response(query, history, tenant_id="default"):
    # 1. Betöltjük a kész keresőt (ami már Hybrid: FAISS + BM25)
    retriever = load_vectorstore(tenant_id)

    if not retriever:
        yield "Hiba: Nincs elérhető tudásbázis. Tölts fel egy PDF-et!"
        return

    # 2. RAG Prompt összeállítása
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Te a SkillSync biztonságos AI asszisztense vagy. Az alábbi kontextus alapján válaszolj:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # 3. A Lánc (Chain) összeállítása LCEL-lel
    chain = (
        {
            "context": (lambda x: x["input"]) | retriever | format_docs, 
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 4. Streaming futtatás
    for chunk in chain.stream({"input": query, "chat_history": history}):
        yield chunk

def get_response(query, history=None, tenant_id="default"):
    """Nem streaming változat a kompatibilitás kedvéért."""
    if history is None: history = []
    full_response = ""
    for chunk in get_streaming_response(query, history, tenant_id):
        full_response += chunk
    
    # Itt most egyszerűsítve adjuk vissza a forrásokat (MVP szint)
    return full_response, ["Helyi PDF dokumentum"]