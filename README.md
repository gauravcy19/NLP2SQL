# 🧠 NLP2SQL – Natural Language to SQL using RAG + LangChain

This project translates natural language questions into executable SQL queries using a Retrieval-Augmented Generation (RAG) approach built on top of LangChain. It enables users to query a database like *"Which artist has the most albums?"* and get the correct SQL and results without writing code.

## 🔍 Key Features

* 💬 **Natural Language Input**: Ask questions in plain English.
* 🔗 **RAG with LangChain**: Uses documents + LLM for better context.
* 📦 **Chroma DB + Embeddings**: Vector store to retrieve relevant schema/context.
* 🛠️ **SQLite**: Sample database (Chinook) for demo purposes.
* 🔐 **.env Support**: Secrets like Hugging Face API keys are handled securely.

## 📁 Structure

* `app.py` – Main entry point to run the app
* `rag_chain.py` – LangChain RAG logic for prompt construction
* `create_embeddings.py` – Embeds DB schema into vector DB
* `prepare_schema.py` – Parses and processes schema
* `chroma_db/` – Local vector database
* `chinook.db` – SQLite sample database
* `.env` – Store Hugging Face token (not committed)
