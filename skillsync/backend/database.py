import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
# Ezek a community csomagok stabilak
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain.retrievers import EnsembleRetriever
except (ImportError, ModuleNotFoundError):
    from langchain_classic.retrievers import EnsembleRetriever

embeddings = OllamaEmbeddings(model="nomic-embed-text")


def load_vectorstore(tenant_id="default"):
    path = f"db/{tenant_id}_faiss"
    if os.path.exists(path):
        # 1. FAISS betöltése
        vectorstore = FAISS.load_local(
            path, embeddings, allow_dangerous_deserialization=True
        )

        # 2. BM25 (kulcsszavas) kereső a dokumentumokból
        docs = list(vectorstore.docstore._dict.values())
        # A logod szerint a community-ből kellene behúzni, próbáljuk meg ott
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 2

        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        # 3. Hybrid kereső
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
        )
        return ensemble_retriever
    return None


def create_or_update_vectorstore(docs, tenant_id="default"):
    # Metaadat tisztítás (9. pont: forrásmegjelölés előkészítése)
    for doc in docs:
        if "source" in doc.metadata:
            # Csak a fájlnevet tároljuk el, ne a teljes elérési utat
            doc.metadata["filename"] = os.path.basename(doc.metadata["source"])

    path = f"db/{tenant_id}_faiss"
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(path)


def rebuild_index_for_tenant(tenant_id="default"):
    upload_dir = f"storage/{tenant_id}"
    all_docs = []

    if os.path.exists(upload_dir):
        files = [f for f in os.listdir(upload_dir) if f.endswith(".pdf")]

        for filename in files:
            file_path = os.path.join(upload_dir, filename)
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            # Metaadat tisztítás (mint az uploadnál)
            for doc in docs:
                doc.metadata["filename"] = filename

            all_docs.extend(docs)

    if all_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        splits = text_splitter.split_documents(all_docs)
        create_or_update_vectorstore(splits, tenant_id)
    else:
        # Ha nem maradt fájl, töröljük az index mappát is
        path = f"db/{tenant_id}_faiss"
        if os.path.exists(path):
            shutil.rmtree(path)
