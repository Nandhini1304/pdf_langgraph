import os
import json
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.vectorstores import FAISS
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_aws.chat_models import ChatBedrock
from langchain_core.prompts import PromptTemplate

# ------------------ Load environment variables ------------------
load_dotenv()

# ------------------ FAISS Vector Store ------------------
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
)
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# ------------------ LLM Setup ------------------
llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant. Use the following context to answer the question clearly.

Context:
{context}

Question:
{question}

Answer:
"""
)

# ------------------ LangGraph State ------------------
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    context: str
    answer: str

# ------------------ Nodes ------------------

def question_node(state: GraphState) -> GraphState:
    last_user_msg = state['messages'][-1].content
    return {**state, "question": last_user_msg}

def retrieve_node(state: GraphState) -> GraphState:
    docs = retriever.invoke(state["question"])
    context = "\n\n".join(doc.page_content for doc in docs)
    return {**state, "context": context}

def generate_node(state: GraphState) -> GraphState:
    prompt = prompt_template.format(context=state["context"], question=state["question"])
    response = llm.invoke(prompt)
    return {**state, "answer": response.content}

def response_node(state: GraphState) -> GraphState:
    return {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": state["answer"]}]
    }

# ------------------ LangGraph Flow ------------------
builder = StateGraph(GraphState)

builder.add_node("question_node", question_node)
builder.add_node("retrieve_node", retrieve_node)
builder.add_node("generate_node", generate_node)
builder.add_node("response_node", response_node)

builder.add_edge(START, "question_node")
builder.set_entry_point("question_node")
builder.add_edge("question_node", "retrieve_node")
builder.add_edge("retrieve_node", "generate_node")
builder.add_edge("generate_node", "response_node")
builder.add_edge("response_node", END)

graph = builder.compile()
# ------------------ Chat Loop ------------------
print("\n LangGraph Chatbot Ready. Type 'exit' to quit.")

state = None
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print(" Goodbye!")
        break

    if state is None:
        state = {
            "messages": [{"role": "user", "content": user_input}],
            "question": "",
            "context": "",
            "answer": ""
        }
    else:
        state["messages"].append({"role": "user", "content": user_input})

    state = graph.invoke(state)
    structured_output = {
        "question": state["question"],
        "answer": state["answer"]
    }

    print("\n Structured JSON Output:")
    print(json.dumps(structured_output, indent=2))