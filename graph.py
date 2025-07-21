import os
import json
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

from huggingface_hub import InferenceClient

# ------------------ Load environment variables ------------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

# ------------------ Load FAISS Vectorstore ------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# ------------------ Hugging Face Chat Model ------------------
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta", token=HF_TOKEN)

def hf_llm_chat(prompt: str) -> str:
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        response = client.chat_completion(messages=messages, max_tokens=512)
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ HF Error:", e)
        return "Error generating response."

# ------------------ Prompt Template ------------------
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Use the following context to answer the question clearly.

Context:
{context}

Question:
{question}

Answer:"""  # <- Ensures model doesn't echo back prompt
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
    print("📨 Prompt to HF:\n", prompt)
    response = hf_llm_chat(prompt)
    print("📥 HF Response:\n", response)
    return {**state, "answer": response or "No response."}

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

builder.set_entry_point("question_node")
builder.add_edge(START, "question_node")
builder.add_edge("question_node", "retrieve_node")
builder.add_edge("retrieve_node", "generate_node")
builder.add_edge("generate_node", "response_node")
builder.add_edge("response_node", END)

graph = builder.compile()

# ------------------ CLI Interface ------------------
if __name__ == "__main__":
    print("\n🤖 Zephyr Chatbot Ready. Type 'exit' to quit.")
    state = None

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
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