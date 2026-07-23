"""
Simple Local RAG-based Data Analyst
Requirements:
    pip install langchain langchain-community langchain-ollama faiss-cpu sentence-transformers pandas
    Install Ollama: https://ollama.com/download
    Then run: ollama pull llama3.2
"""

from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

CSV_FILE = "synthetic_sales_data.csv"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "gemma2:2b"  # change to "mistral" or any model you have pulled

# --- Load CSV ---
print("Loading data...")
loader = CSVLoader(file_path=CSV_FILE)
docs = loader.load()
print(f"  Loaded {len(docs)} rows")

# --- Embeddings & Vector Store ---
print("Building vector store...")
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# --- LLM (local via Ollama) ---
llm = OllamaLLM(model=OLLAMA_MODEL)

# --- Query Loop ---
print("\nRAG Data Analyst ready! Type 'exit' to quit.\n")
while True:
    query = input("Your question: ").strip()
    if query.lower() in ("exit", "quit"):
        break
    if not query:
        continue

    # Retrieve relevant rows
    retrieved = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in retrieved])

    prompt = f"""You are a data analyst. Answer the question using only the CSV data context below.
Be concise and specific. If you can't answer from the context, say so.

Context:
{context}

Question: {query}

Answer:"""

    print("\nAnalyzing...")
    response = llm.invoke(prompt)
    print(f"\n{response}\n")
    print("-" * 50)
