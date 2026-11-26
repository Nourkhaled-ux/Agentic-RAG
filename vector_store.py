from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def build_vector_store(chunks: list):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def get_retriever(vector_store, k: int = 5):
    return vector_store.as_retriever(search_kwargs={"k": k})


def search_documents(vector_store, query: str, k: int = 5) -> list:
    results = vector_store.similarity_search(query, k=k)
    return results