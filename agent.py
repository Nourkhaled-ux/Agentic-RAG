import json
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from tools import search_tool, summarize_tool, calculate_tool, analyze_data_tool, compare_tool


AGENT_SYSTEM_PROMPT = """You are an intelligent Agentic RAG assistant with access to a knowledge base and tools.

Your capabilities:
1. SEARCH: Retrieve relevant documents from the knowledge base
2. SUMMARIZE: Summarize long texts or retrieved content
3. CALCULATE: Perform mathematical calculations
4. ANALYZE: Analyze structured data
5. COMPARE: Compare information from multiple sources

Your reasoning process:
1. UNDERSTAND: Analyze the user question deeply
2. PLAN: Decide which tools to use and in what order
3. EXECUTE: Use tools to gather information
4. SYNTHESIZE: Combine findings into a coherent answer
5. CITE: Always cite your sources

Always respond ONLY with this exact JSON format and nothing else:
{
  "thought": "Your reasoning about the question",
  "plan": ["step1", "step2"],
  "tool_calls": [
    {"tool": "tool_name", "params": {}, "reason": "why this tool"}
  ],
  "final_answer": "Your comprehensive answer",
  "sources": ["source1", "source2"]
}

Available tools: search, summarize, calculate, analyze_data, compare

Be thorough, accurate, and always cite sources from the knowledge base."""


class AgenticRAG:
    def __init__(self, vector_store, groq_api_key: str, model: str):
        self.vector_store = vector_store
        self.groq_api_key = groq_api_key
        self.model = model
        self.llm = ChatGroq(
            model=model,
            groq_api_key=groq_api_key,
            temperature=0,
        )
        self.conversation_history = []
        self.tool_execution_log = []

    def plan(self, question: str) -> dict:
        messages = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            *self.conversation_history,
            HumanMessage(content=f"Question: {question}")
        ]

        response = self.llm.invoke(messages)

        raw = response.content.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]

        return json.loads(raw.strip())

    def execute_tool(self, tool_name: str, params: dict) -> dict:
        if tool_name == "search":
            result = search_tool(params.get("query", ""), self.vector_store, params.get("k", 5))

        elif tool_name == "summarize":
            result = summarize_tool(params.get("text", ""), self.groq_api_key, self.model, params.get("focus", None))

        elif tool_name == "calculate":
            result = calculate_tool(params.get("expression", ""))

        elif tool_name == "analyze_data":
            result = analyze_data_tool(params.get("data", []), params.get("analysis_type", "summary"))

        elif tool_name == "compare":
            result = compare_tool(params.get("query", ""), self.vector_store, self.groq_api_key, self.model, params.get("k", 6))

        else:
            result = {"status": "error", "message": f"Unknown tool: {tool_name}"}

        self.tool_execution_log.append({"tool": tool_name, "params": params, "result": result})
        return result

    def refine_answer(self, question: str, initial_plan: dict, tool_results: list) -> str:
        tool_results_text = json.dumps(tool_results, indent=2)

        refinement_prompt = f"""
Based on the following tool results, provide a comprehensive, accurate answer to the question.

Question: {question}

Initial Plan: {json.dumps(initial_plan.get('plan', []))}

Tool Results:
{tool_results_text}

Provide a detailed, well-structured answer that:
1. Directly answers the question
2. Synthesizes information from multiple sources
3. Cites specific sources
4. Highlights any conflicting information
5. Notes any gaps or limitations

Format your response as:
ANSWER: [Your comprehensive answer]
SOURCES: [List of sources used]
CONFIDENCE: [High/Medium/Low with reason]
"""

        messages = [
            SystemMessage(content="You are an expert at synthesizing information and providing accurate, cited answers."),
            HumanMessage(content=refinement_prompt)
        ]

        response = self.llm.invoke(messages)
        return response.content

    def run(self, question: str) -> dict:
        self.tool_execution_log = []

        plan = self.plan(question)

        tool_results = []
        for tool_call in plan.get("tool_calls", []):
            result = self.execute_tool(tool_call.get("tool"), tool_call.get("params", {}))
            tool_results.append({
                "tool": tool_call.get("tool"),
                "params": tool_call.get("params", {}),
                "result": result,
                "reason": tool_call.get("reason", "")
            })

        refined_answer = self.refine_answer(question, plan, tool_results)

        self.conversation_history.append(HumanMessage(content=question))
        self.conversation_history.append(AIMessage(content=refined_answer))

        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        return {
            "question": question,
            "thought": plan.get("thought", ""),
            "plan": plan.get("plan", []),
            "tool_executions": self.tool_execution_log,
            "refined_answer": refined_answer,
            "sources": plan.get("sources", []),
        }

    def reset_conversation(self):
        self.conversation_history = []
        self.tool_execution_log = []