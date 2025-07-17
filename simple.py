import os
import json
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_aws.chat_models import ChatBedrock
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# ------------------ Load environment ------------------
load_dotenv()

# ------------------ Vector store ------------------
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
)
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# ------------------ LLM ------------------
llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
)

# ------------------ Prompt ------------------
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Use the context below to answer the question accurately.

Context:
{context}

Question:
{question}

Answer:"""
)

# ------------------ State ------------------
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    context: str
    answer: str

# ------------------ Tool: human_assistance ------------------
@tool
def human_assistance(query: str) -> str:
    """Ask a human when the system cannot proceed."""
    raise interrupt({"query": query})

tools = [human_assistance]
llm_with_tools = llm.bind_tools(tools)

# ------------------ Nodes ------------------
def question_node(state: GraphState) -> GraphState:
    user_msg = state["messages"][-1].content
    return {**state, "question": user_msg}

def retrieve_node(state: GraphState) -> GraphState:
    docs = retriever.invoke(state["question"])
    if not docs:
        # Instead of direct interrupt, we use tool call
        state["messages"].append(HumanMessage(content="Please help me with this question."))
        tool_call = llm_with_tools.invoke(state["messages"])
        return {"messages": state["messages"] + [tool_call]}
    context = "\n\n".join(doc.page_content for doc in docs)
    return {**state, "context": context}

def generate_node(state: GraphState) -> GraphState:
    prompt = prompt_template.format(context=state["context"], question=state["question"])
    response = llm.invoke(prompt)
    return {**state, "answer": response.content}

def response_node(state: GraphState) -> GraphState:
    return {
        **state,
        "messages": state["messages"] + [HumanMessage(content=state["answer"])]
    }

# ------------------ Build Graph ------------------
builder = StateGraph(GraphState)

builder.add_node("question_node", question_node)
builder.add_node("retrieve_node", retrieve_node)
builder.add_node("generate_node", generate_node)
builder.add_node("response_node", response_node)

# Tool node
tool_node = ToolNode(tools=tools)
builder.add_node("tools", tool_node)

# Routing
builder.set_entry_point("question_node")
builder.add_edge(START, "question_node")
builder.add_edge("question_node", "retrieve_node")
builder.add_edge("retrieve_node", "generate_node")
builder.add_edge("generate_node", "response_node")
builder.add_edge("response_node", END)

# Tool invocation loop
builder.add_conditional_edges("retrieve_node", tools_condition)
builder.add_edge("tools", "generate_node")

graph = builder.compile()

# ------------------ CLI Chat ------------------
def chat():
    print(" LangGraph ")
    print("Type 'exit' to quit.\n")

    state = {
        "messages": [],
        "question": "",
        "context": "",
        "answer": ""
    }

    while True:
        user_input = input("👤 You: ")
        if user_input.lower() in {"exit", "quit"}:
            break

        state["messages"].append(HumanMessage(content=user_input))

        try:
            state = graph.invoke(state)
        except Exception as e:
            if hasattr(e, "type") and e.type == "interrupt":
                print(json.dumps({
                    "status": "human_required",
                    "query": e.data.get("query", "unknown")
                }, indent=2))
                human_response = input(" Human input: ")
                state["context"] = human_response
                state = generate_node(state)
                state = response_node(state)
            else:
                print(" Error:", str(e))
                continue

        print(json.dumps({
            "status": "success",
            "question": state["question"],
            "answer": state["answer"]
        }, indent=2))

if __name__ == "__main__":
    chat()
