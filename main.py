from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_aws.embeddings import BedrockEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

# ------------------ Load PDF ------------------
loader = PyPDFLoader("D:\python\Mango_doc.pdf")  # make sure this file exists
docs = loader.load()

# ------------------ Split Text ------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# ------------------ Embeddings ------------------
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
)

# ------------------ Save FAISS Index ------------------
vectorstore = FAISS.from_documents(split_docs, embeddings)
vectorstore.save_local("faiss_index")

print("✅ FAISS index saved.")
