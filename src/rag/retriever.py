import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

import pickle

INDEX_PATH = "./faiss_cache"
DOCS_PATH = "./docs_cache.pkl"

def get_documents():
    if os.path.exists(DOCS_PATH):
        print("Loading documents from cache...")
        with open(DOCS_PATH, "rb") as f:
            return pickle.load(f)
            
    print("Loading documents from URLs...")
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(doc_splits, f)
    return doc_splits

def get_retriever():
    embedding_model = HuggingFaceEmbeddings()
    
    # 1. FAISS Vector Store (Semantic Search)
    if os.path.exists(INDEX_PATH):
        print("Loading FAISS from cache...")
        vectorstore = FAISS.load_local(INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
    else:
        print("Creating new FAISS vector store...")
        doc_splits = get_documents()
        vectorstore = FAISS.from_documents(documents=doc_splits, embedding=embedding_model)
        vectorstore.save_local(INDEX_PATH)
        
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 2. BM25 Retriever (Keyword Search)
    # We load docs again if cache hit, just for BM25 (in a real app, cache the docs too)
    # But for simplicity, we'll fast-load if needed
    print("Initializing BM25 Retriever...")
    doc_splits = get_documents() # In memory fast enough if already loaded, but wait, WebBaseLoader is slow.
    # Actually let's just do it directly. In a prod app, doc_splits would be saved to disk.
    
    bm25_retriever = BM25Retriever.from_documents(doc_splits)
    bm25_retriever.k = 3
    
    # 3. Ensemble Retriever
    print("Creating Ensemble Hybrid Retriever...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.4, 0.6]
    )
    
    return ensemble_retriever
