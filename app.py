import os
import streamlit as st
from dotenv import load_dotenv
from pipeline import run_pipeline
from vector_store import build_vector_store
from agent import AgenticRAG

load_dotenv()

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0f1117;
    --bg-secondary: #1a1d2e;
    --bg-card: #1e2235;
    --accent-primary: #6366f1;
    --accent-secondary: #818cf8;
    --accent-success: #34d399;
    --accent-warning: #fbbf24;
    --accent-error: #f87171;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --border: #2d3154;
    --border-active: #6366f1;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary);
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

[data-testid="stSidebar"] {
    background-color: var(--bg-secondary);
    border-right: 1px solid var(--border);
}

.main-header {
    background: linear-gradient(135deg, #1e2235 0%, #252840 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #6366f1, #818cf8, #34d399);
}

.main-header h1 {
    font-size: 28px;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}

.main-header p {
    color: var(--text-secondary);
    font-size: 14px;
    margin: 0;
}

.thought-box {
    background: linear-gradient(135deg, #1e2235, #252840);
    border: 1px solid var(--accent-primary);
    border-left: 3px solid var(--accent-primary);
    border-radius: 10px;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 14px;
    color: var(--text-secondary);
    font-style: italic;
}

.plan-step {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 16px;
    margin: 6px 0;
    font-size: 13px;
    color: var(--text-secondary);
}

.tool-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 13px;
}

.tool-badge {
    background: var(--accent-primary);
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
}

.answer-box {
    background: linear-gradient(135deg, #1e2235, #1a2940);
    border: 1px solid var(--accent-success);
    border-left: 3px solid var(--accent-success);
    border-radius: 12px;
    padding: 24px 28px;
    margin: 16px 0;
    color: var(--text-primary);
    line-height: 1.7;
    font-size: 15px;
    white-space: pre-wrap;
}

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}

.metric-card .metric-value {
    font-size: 28px;
    font-weight: 700;
    color: var(--accent-primary);
    font-family: 'JetBrains Mono', monospace;
}

.metric-card .metric-label {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    margin-bottom: 10px;
}

.status-ready {
    background: rgba(52, 211, 153, 0.15);
    color: var(--accent-success);
    border: 1px solid rgba(52, 211, 153, 0.3);
}

.status-not-ready {
    background: rgba(248, 113, 113, 0.15);
    color: var(--accent-error);
    border: 1px solid rgba(248, 113, 113, 0.3);
}

.chat-user {
    background: linear-gradient(135deg, #2a2d4a, #252840);
    border: 1px solid var(--border);
    border-radius: 12px 12px 4px 12px;
    padding: 16px 20px;
    margin: 12px 0;
    color: var(--text-primary);
    font-size: 15px;
}

.chat-assistant {
    background: linear-gradient(135deg, #1e2235, #1e2638);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 12px 12px 12px 4px;
    padding: 16px 20px;
    margin: 12px 0;
    color: var(--text-primary);
    font-size: 15px;
}

.sidebar-section {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    margin: 10px 0;
}

.sidebar-section h4 {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 0 0 12px 0;
}

.groq-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(129,140,248,0.2));
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 600;
    color: var(--accent-secondary);
    margin-bottom: 12px;
}

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] > div > div {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
}

.stButton button {
    background: linear-gradient(135deg, #6366f1, #818cf8) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
}

.stExpander {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-primary); }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    defaults = {
        "vector_store": None,
        "agent": None,
        "chat_history": [],
        "kb_built": False,
        "groq_api_key": "",
        "selected_model": GROQ_MODELS[0],
        "total_queries": 0,
        "total_chunks": 0,
        "kb_folder": "knowledge_base",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding: 8px 0 20px 0;">
            <div style="font-size: 20px; font-weight: 700; color: #f1f5f9;">🧠 Agentic RAG</div>
            <div style="font-size: 12px; color: #64748b; margin-top: 4px;">Powered by Groq</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section"><h4>Groq Configuration</h4>', unsafe_allow_html=True)

        st.markdown('<div class="groq-badge">⚡ Groq Inference</div>', unsafe_allow_html=True)

        api_key_input = st.text_input(
            "Groq API Key",
            value=st.session_state.groq_api_key or os.getenv("GROQ_API_KEY", ""),
            type="password",
            placeholder="gsk_...",
            label_visibility="collapsed"
        )
        if api_key_input:
            st.session_state.groq_api_key = api_key_input

        selected_model = st.selectbox(
            "Model",
            options=GROQ_MODELS,
            index=GROQ_MODELS.index(st.session_state.selected_model),
            label_visibility="collapsed"
        )
        st.session_state.selected_model = selected_model

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section"><h4>Pipeline Settings</h4>', unsafe_allow_html=True)

        kb_folder = st.text_input(
            "Knowledge Base Folder",
            value=st.session_state.kb_folder,
            placeholder="knowledge_base",
            label_visibility="collapsed"
        )
        st.session_state.kb_folder = kb_folder

        chunk_size = st.slider("Chunk Size", 200, 1000, 500, 50)
        retrieval_k = st.slider("Top-K Retrieval", 1, 10, 5)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section"><h4>Knowledge Base</h4>', unsafe_allow_html=True)

        status_class = "status-ready" if st.session_state.kb_built else "status-not-ready"
        status_text = "● Ready" if st.session_state.kb_built else "● Not Built"
        st.markdown(f'<div class="status-pill {status_class}">{status_text}</div>', unsafe_allow_html=True)

        if st.button("Build Knowledge Base", use_container_width=True):
            if not st.session_state.groq_api_key:
                st.error("Please enter your Groq API Key first.")
            elif not os.path.exists(st.session_state.kb_folder):
                st.error(f"Folder '{st.session_state.kb_folder}' not found.")
            else:
                with st.spinner("Processing documents..."):
                    chunks = run_pipeline(st.session_state.kb_folder)
                    if not chunks:
                        st.error("No documents found in the folder.")
                    else:
                        vector_store = build_vector_store(chunks)
                        st.session_state.vector_store = vector_store
                        st.session_state.total_chunks = len(chunks)
                        st.session_state.agent = AgenticRAG(
                            vector_store,
                            st.session_state.groq_api_key,
                            st.session_state.selected_model
                        )
                        st.session_state.kb_built = True
                        st.success(f"Built from {len(chunks)} chunks!")
                        st.rerun()

        if st.session_state.kb_built:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{st.session_state.total_chunks}</div>
                    <div class="metric-label">Chunks</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{st.session_state.total_queries}</div>
                    <div class="metric-label">Queries</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.chat_history:
            st.markdown('<div class="sidebar-section"><h4>Session</h4>', unsafe_allow_html=True)
            if st.button("Clear History", use_container_width=True):
                st.session_state.chat_history = []
                if st.session_state.agent:
                    st.session_state.agent.reset_conversation()
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section"><h4>Available Tools</h4>', unsafe_allow_html=True)
        tools = [
            ("🔍", "Search", "Semantic document retrieval"),
            ("📝", "Summarize", "Text condensation"),
            ("🧮", "Calculate", "Math expressions"),
            ("📊", "Analyze", "Data statistics"),
            ("⚖️", "Compare", "Multi-source comparison"),
        ]
        for icon, name, desc in tools:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid var(--border);">
                <span style="font-size:16px;">{icon}</span>
                <div>
                    <div style="font-size:13px;font-weight:500;color:var(--text-primary);">{name}</div>
                    <div style="font-size:11px;color:var(--text-muted);">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        return retrieval_k


def render_response(result: dict):
    if result.get("thought"):
        with st.expander("Agent Reasoning", expanded=False):
            st.markdown(f'<div class="thought-box">💭 {result["thought"]}</div>', unsafe_allow_html=True)

    if result.get("plan"):
        with st.expander("Execution Plan", expanded=False):
            for i, step in enumerate(result["plan"], 1):
                st.markdown(f"""
                <div class="plan-step">
                    <strong style="color:var(--accent-primary);">{i}.</strong> {step}
                </div>
                """, unsafe_allow_html=True)

    if result.get("tool_executions"):
        with st.expander(f"Tool Usage ({len(result['tool_executions'])} calls)", expanded=False):
            tool_icons = {
                "search": "🔍", "summarize": "📝",
                "calculate": "🧮", "analyze_data": "📊", "compare": "⚖️"
            }
            for exec_item in result["tool_executions"]:
                icon = tool_icons.get(exec_item["tool"], "🔧")
                status = exec_item["result"].get("status", "unknown")
                status_color = "#34d399" if status == "success" else "#f87171"
                st.markdown(f"""
                <div class="tool-card">
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                        <span>{icon}</span>
                        <span class="tool-badge">{exec_item['tool']}</span>
                        <span style="color:{status_color};font-size:11px;font-weight:500;">● {status.upper()}</span>
                    </div>
                    <div style="color:var(--text-muted);font-size:12px;">
                        Params: <code style="font-family:'JetBrains Mono';color:var(--accent-secondary);">{str(exec_item['params'])[:120]}</code>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    if result.get("refined_answer"):
        st.markdown(f'<div class="answer-box">{result["refined_answer"]}</div>', unsafe_allow_html=True)


def main():
    initialize_session_state()
    render_sidebar()

    st.markdown("""
    <div class="main-header">
        <h1>🧠 Agentic RAG System</h1>
        <p>Powered by Groq — Ask complex, multi-step questions. The agent plans, retrieves, reasons, and answers with citations.</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.kb_built:
        st.markdown("""
        <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:40px;text-align:center;margin:40px 0;">
            <div style="font-size:48px;margin-bottom:16px;">📚</div>
            <div style="font-size:20px;font-weight:600;color:var(--text-primary);margin-bottom:8px;">
                Knowledge Base Not Ready
            </div>
            <div style="font-size:14px;color:var(--text-secondary);max-width:420px;margin:0 auto;">
                Add your PDF or TXT files to the <code>knowledge_base/</code> folder,
                enter your Groq API key in the sidebar, then click Build Knowledge Base.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    for entry in st.session_state.chat_history:
        if entry["role"] == "user":
            st.markdown(f'<div class="chat-user"><strong style="color:var(--accent-secondary);">You</strong><br><br>{entry["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="chat-assistant"><strong style="color:var(--accent-success);">Agent</strong></div>', unsafe_allow_html=True)
            render_response(entry["content"])

    st.markdown("---")

    example_questions = [
        "What are the main topics covered in the documents?",
        "Compare and contrast the key ideas across all sources.",
        "Summarize the most important findings with citations.",
        "What relationships exist between the different concepts?",
    ]

    with st.expander("Example Questions", expanded=False):
        cols = st.columns(2)
        for i, q in enumerate(example_questions):
            with cols[i % 2]:
                if st.button(q, key=f"example_{i}", use_container_width=True):
                    st.session_state["prefill_question"] = q

    question = st.text_area(
        "Your Question",
        value=st.session_state.pop("prefill_question", ""),
        placeholder="Ask a complex, multi-step question about your documents...",
        height=100,
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([4, 1])
    with col1:
        submit = st.button("Send to Agent →", use_container_width=True)

    if submit and question.strip():
        st.session_state.chat_history.append({"role": "user", "content": question.strip()})
        st.session_state.total_queries += 1

        with st.spinner("⚡ Groq is thinking, planning, and retrieving..."):
            result = st.session_state.agent.run(question.strip())

        st.session_state.chat_history.append({"role": "assistant", "content": result})
        st.rerun()

    elif submit and not question.strip():
        st.warning("Please enter a question.")


if __name__ == "__main__":
    main()