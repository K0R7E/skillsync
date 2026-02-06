import os
# Ezek a community csomagok stabilak
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever

try:
    from langchain.retrievers import EnsembleRetriever
except (ImportError, ModuleNotFoundError):
    from langchain_classic.retrievers import EnsembleRetriever

embeddings = OllamaEmbeddings(model="nomic-embed-text")

def load_vectorstore(tenant_id="default"):
    path = f"db/{tenant_id}_faiss"
    if os.path.exists(path):
        # 1. FAISS betöltése
        vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        
        # 2. BM25 (kulcsszavas) kereső a dokumentumokból
        docs = list(vectorstore.docstore._dict.values())
        # A logod szerint a community-ből kellene behúzni, próbáljuk meg ott
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 2 
        
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        
        # 3. Hybrid kereső
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )
        return ensemble_retriever
    return None

def create_or_update_vectorstore(docs, tenant_id="default"):
    path = f"db/{tenant_id}_faiss"
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(path)
    return vectorstore