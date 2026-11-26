import re
import math
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


def search_tool(query: str, vector_store, k: int = 5) -> dict:
    results = vector_store.similarity_search(query, k=k)

    if not results:
        return {"status": "no_results", "data": [], "query": query}

    formatted = []
    for i, doc in enumerate(results):
        formatted.append({
            "chunk_id": i + 1,
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "N/A"),
        })

    return {"status": "success", "data": formatted, "query": query}


def summarize_tool(text: str, groq_api_key: str, model: str, focus: str = None) -> dict:
    llm = ChatGroq(model=model, groq_api_key=groq_api_key, temperature=0)

    system_prompt = "You are an expert summarizer. Provide concise, accurate summaries."
    if focus:
        system_prompt += f" Focus specifically on: {focus}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Summarize the following text:\n\n{text}")
    ]

    response = llm.invoke(messages)
    return {"status": "success", "summary": response.content}


def calculate_tool(expression: str) -> dict:
    expression = expression.strip()

    allowed_names = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow, "sqrt": math.sqrt,
        "log": math.log, "log10": math.log10, "pi": math.pi, "e": math.e,
        "ceil": math.ceil, "floor": math.floor,
    }

    cleaned = re.sub(r'[^0-9+\-*/().%\s,a-zA-Z_]', '', expression)
    result = eval(cleaned, {"__builtins__": {}}, allowed_names)
    return {"status": "success", "expression": expression, "result": result}


def analyze_data_tool(data: list, analysis_type: str = "summary") -> dict:
    if not data:
        return {"status": "error", "message": "No data provided"}

    df = pd.DataFrame(data) if isinstance(data[0], dict) else pd.DataFrame({"values": data})

    if analysis_type == "summary":
        return {"status": "success", "analysis_type": analysis_type, "result": df.describe().to_dict()}
    elif analysis_type == "count":
        return {"status": "success", "analysis_type": analysis_type, "result": df.count().to_dict()}
    elif analysis_type == "unique":
        return {"status": "success", "analysis_type": analysis_type, "result": {col: df[col].nunique() for col in df.columns}}

    return {"status": "error", "message": f"Unknown analysis type: {analysis_type}"}


def compare_tool(query: str, vector_store, groq_api_key: str, model: str, k: int = 6) -> dict:
    results = vector_store.similarity_search(query, k=k)

    if not results:
        return {"status": "no_results", "comparison": ""}

    combined_text = "\n\n---\n\n".join([
        f"Source {i+1} ({doc.metadata.get('source', 'Unknown')}):\n{doc.page_content}"
        for i, doc in enumerate(results)
    ])

    llm = ChatGroq(model=model, groq_api_key=groq_api_key, temperature=0)

    messages = [
        SystemMessage(content="You are an expert at comparing and contrasting information from multiple sources."),
        HumanMessage(content=f"Compare the following information regarding: {query}\n\n{combined_text}")
    ]

    response = llm.invoke(messages)
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in results]))
    return {"status": "success", "comparison": response.content, "sources": sources}