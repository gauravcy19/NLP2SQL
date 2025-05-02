import sqlite_utils
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# --- Function to extract schema text (same as in prepare_schema.py) ---
def get_schema_text(db_path):
    db = sqlite_utils.Database(db_path)
    schema_text = ""
    for table in db.tables:
        schema_text += f"Table: {table.name}\n"
        schema_text += table.schema + "\n\n"
    return schema_text

# --- Database Path Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
db_path = os.path.join(script_dir, "chinook.db") # Full path

# --- Embedding Model ---
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 

# --- Text Splitter ---
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# --- Load schema and split into chunks ---
schema_string = get_schema_text(db_path)
texts = text_splitter.split_text(schema_string)

# --- Create ChromaDB vector store ---
db = Chroma.from_texts(texts, embedding_function, persist_directory="chroma_db")
db.persist()  # Save the database to disk

print("ChromaDB embeddings created and stored in 'chroma_db' directory.")