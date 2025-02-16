import os
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings

# Set local directories
DOCUMENTS_FOLDER = "rag_documents"
VECTOR_STORE_FOLDER = "rag_vector_store"

os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_FOLDER, exist_ok=True)

def load_documents(directory=DOCUMENTS_FOLDER):
    """Load text documents as whole documents and return as LangChain Document objects."""
    docs = []
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            try:
                file_path = os.path.join(directory, file)
                loader = TextLoader(file_path, encoding="utf-8")
                
                # Load the document and extract text
                loaded_docs = loader.load()  # This returns a list
                
                if isinstance(loaded_docs, list) and len(loaded_docs) > 0:
                    document_content = loaded_docs[0].page_content  # Extract the text content
                    
                    metadata = {"source": file}  # Store filename for metadata
                    
                    # Convert to LangChain Document object
                    docs.append(Document(page_content=document_content, metadata=metadata))
                else:
                    print(f"⚠️ Warning: {file} was loaded but is empty.")

            except Exception as e:
                print(f"⚠️ Error loading {file}: {e}")
    return docs

def chunk_documents(documents):
    """Split documents into smaller chunks for better retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Adjusted for better retrieval
        chunk_overlap=300  # Maintain context
    )
    return text_splitter.split_documents(documents)

def store_embeddings(docs, db_folder=VECTOR_STORE_FOLDER):
    """Store document embeddings in FAISS."""
    embeddings = OllamaEmbeddings(model="deepseek-r1:14b")
    
    # Extract text and metadata from Document objects
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    vector_store.save_local(db_folder)
    return vector_store

def load_vector_store(db_folder=VECTOR_STORE_FOLDER):
    """Load the FAISS vector store with adjusted search settings."""
    embeddings = OllamaEmbeddings(model="deepseek-r1:14b")

    vector_store = FAISS.load_local(db_folder, embeddings, allow_dangerous_deserialization=True)
    
    # Adjust FAISS settings for better retrieval
    vector_store.search_kwargs = {"k": 5, "fetch_k": 10}  # More diverse retrieval

    return vector_store


def create_rag_pipeline(vector_store):
    """Retrieve whole race summaries and ensure direct answers."""
    llm = OllamaLLM(model="deepseek-r1:14b")

    retriever = vector_store.as_retriever(
        # Use Maximal Marginal Relevance (better ranking)
        search_type="mmr",
        # Retrieve whole documents, not small fragments  
        search_kwargs={"k": 5, "fetch_k": 10}  
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Use the following race summaries to answer the question concisely:\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Provide a direct answer. If the answer is unknown, return 'Not available'."
        )
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

if __name__ == "__main__":
    index_path = os.path.join(VECTOR_STORE_FOLDER, "index.faiss")
    if os.path.exists(index_path):
        print("Loading existing FAISS vector store...")
        vector_store = load_vector_store()
    else:
        print("No FAISS index found. Creating a new one...")
        documents = load_documents()
        if not documents:
            print("⚠️ No documents found in 'rag_documents'. Add .txt files and try again.")
            exit()
        chunks = chunk_documents(documents)
        vector_store = store_embeddings(chunks)

    rag_pipeline = create_rag_pipeline(vector_store)

    while True:
        query = input("\nAsk a question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        response = rag_pipeline.invoke(query)

        # Extract relevant parts from the response
        if isinstance(response, dict):
            query_text = response.get("query", "N/A")
            result_text = response.get("result", "No response available").strip()

            # Clean up unnecessary tags and internal thoughts (if any)
            formatted_result = result_text.replace("<think>", "").replace("</think>", "").strip()

            # Display response in a human-readable way
            print("\n🔍 **Query:**", query_text)
            print("\n💡 **Response:**\n", formatted_result)
        else:
            print("\n💡 Response:", response)