import os
import sqlite_utils
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.retrieval_qa.base import BaseRetrievalQA

# --- Load environment variables ---
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "API Token" 

# --- LLM Configuration ---
llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.1", model_kwargs={"temperature":0.1, "max_length":2048}) #use llama2-7b-chat-hf or Mistral-7B-Instruct-v0.1

# --- Prompt Template ---
prompt_template = """You are a SQL expert. Given an input question, first retrieve relevant tables from the database schema. Then, generate a syntactically correct SQLite query to answer the question. Only use the tables and columns provided in the database schema.

Question: {question}

Database Schema: {context}

SQL Query:"""

prompt = PromptTemplate.from_template(prompt_template)

# --- Embedding Model ---
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # Use a small, efficient model

# --- Load ChromaDB ---
db = Chroma(persist_directory="chroma_db", embedding_function=embedding_function)
retriever = db.as_retriever(search_kwargs={"k": 2})  # Adjust 'k' as needed (number of chunks to retrieve)

# --- Chain the RAG components ---
llm_chain = LLMChain(prompt=prompt, llm=llm)
combine_documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain, document_variable_name="context"
)

rag_chain = RetrievalQA(
    retriever=retriever, combine_documents_chain=combine_documents_chain
)

def get_response(rag_chain, query):
    response = rag_chain.run(query)
    return response

# Example Usage:
query = "Find all rock artists"
response = get_response(rag_chain, query)

print("Generated SQL:", response)