import os
import re
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(folder_path: str) -> list:
    documents = []
    folder = Path(folder_path)

    for file_path in folder.iterdir():
        if file_path.suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())
        elif file_path.suffix in [".txt", ".md"]:
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents.extend(loader.load())

    return documents


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    return text


def chunk_documents(documents: list, chunk_size: int = 500, chunk_overlap: int = 100) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = []
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
        chunks.extend(splitter.split_documents([doc]))

    return chunks


def run_pipeline(folder_path: str) -> list:
    documents = load_documents(folder_path)
    chunks = chunk_documents(documents)
    return chunks