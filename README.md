# RAG-based Retrieval System

This project is a **test implementation** of a **Retrieval-Augmented Generation (RAG)** pipeline using **LangChain**, **FAISS**, and **Ollama** for efficient document retrieval and question answering. The primary goal is to experiment with building a simple RAG system locally. 

## ✨ Features

- Processes `.txt` documents for intelligent retrieval.
- Splits text into chunks for better retrieval.
- Stores & retrieves embeddings using FAISS.
- Uses `deepseek-r1:14b` model for answering queries.
- Utilizes Maximal Marginal Relevance (MMR) ranking to enhance retrieval relevance.
- Running seamlessly on a local environment

## 🚀 Installation

### Prerequisites

Before getting started, ensure you have the following:

- Python 3.12+
- Necessary dependencies

### Install dependencies

```sh
pip install langchain langchain-community faiss-cpu ollama
```

## 📖 Usage

### Step 1: Prepare Your Documents

Place your `.txt` files containing structured documents inside the `rag_documents/` folder.

### Step 2: Launch the System

```sh
python rag_retrieval.py
```

If a FAISS vector store is detected, it will be loaded automatically. Otherwise, the system will generate a new index from the available documents.

### Step 3: Ask Your Questions

Once the script is running, input your queries:

```sh
Ask a question: "Who won the Dutch Grand Prix?"
```

The system will analyze relevant document segments and generate an informed response.

### Exiting the Program

Simply type `exit` to close the interactive session.

## 📂 Project Structure

```
|-- rag_documents/      # Folder containing input text documents
|-- rag_vector_store/   # Directory housing FAISS index and embeddings
|-- rag_retrieval.py    # Core script
|-- README.md           # Project documentation
```
## 🏎 Test Data

This project uses Wikipedia summaries of the 2024 F1 season as test data. The `.txt` files contain structured race summaries, which are processed for retrieval and question answering.

## 🔍 Notes

- Ensure `.txt` files in `rag_documents/` contain structured and meaningful information.
- If no FAISS index is found, the system automatically processes available documents.
- This project harnesses the power of the `deepseek-r1:14b` model for embeddings and retrieval.

## 🔮 Future Work

Future work will focus on improving retrieval accuracy and optimizing performance.
