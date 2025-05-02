import streamlit as st
import pandas as pd
import sqlite_utils
import os
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# --- Load environment variables ---
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "API Token" 


@st.cache_resource()  # cache the db, llm and retriever
def load_stuff():
    # --- Database Schema Loading ---
    def get_schema_text(db_path):
        db = sqlite_utils.Database(db_path)
        schema_text = ""
        for table in db.tables:
            schema_text += f"Table: {table.name}\n"
            schema_text += table.schema + "\n\n"
        return schema_text

    # --- Database Path Configuration ---
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
    db_path = os.path.join(script_dir, "chinook.db")  # Full path
    schema_string = get_schema_text(db_path)

    # --- Embedding and Vector Store Creation ---
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Use a small, efficient model
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text(schema_string)
    db = Chroma.from_texts(texts, embedding_function, persist_directory="chroma_db")
    db.persist()  # Save the database to disk
    retriever = db.as_retriever(search_kwargs={"k": 1})  # Adjust 'k' as needed (number of chunks to retrieve)

    # --- LLM Configuration ---
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        model_kwargs={"temperature": 0.1, "max_length": 2048, "stop": ["\n\n", ";"]},
    )  # use llama2-7b-chat-hf or Mistral-7B-Instruct-v0.1
    # --- Prompt Template ---
    prompt_template = """You are a SQL expert. You are given a question about a music database. You must generate a syntactically correct SQLite query to answer the question.
    Here are some rules for generating the SQL query:
    - Only use the tables and columns provided in the database schema.
    - You should join tables based on the foreign key relationships, if required
    - You MUST only output one SQL query, do not add any additional text, comments, or explanations. End the query with a semicolon (;).

    Here is the database schema:
    {context}

    Question: {question}

    SQL Query: """

    prompt = PromptTemplate.from_template(prompt_template)

    rag_chain = (
        {"context": retriever | RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


rag_chain = load_stuff()


# --- SQL Execution Function ---
def execute_sql(db_path, sql_query):
    """Executes an SQL query against a SQLite database and returns the results as a Pandas DataFrame."""
    try:
        db = sqlite_utils.Database(db_path)
        results = db.query(sql_query)  # Returns an iterator of rows
        df = pd.DataFrame(results)  # Convert to Pandas DataFrame
        return df
    except Exception as e:
        return None  # Return None if it fails


def get_response(rag_chain, query):
    response = rag_chain.invoke(query)
    return response


# --- Streamlit UI ---
st.title("Chinook Database Text-2-SQL POC")
query = st.text_input("Enter your query:")

if query:
    st.markdown(f"**Question:** {query}")
    generated_sql = rag_chain.invoke(query)
    st.code(generated_sql, language="sql")

    # --- Database Path Configuration ---
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
    db_path = os.path.join(script_dir, "chinook.db")  # Full path
    df_results = execute_sql(db_path, generated_sql)
    if df_results is not None:  # check that it is not none
        st.write("SQL Execution Results:")
        st.dataframe(df_results)
    else:
        st.write("There was an error in the SQL statement, confirm that the query is valid.")