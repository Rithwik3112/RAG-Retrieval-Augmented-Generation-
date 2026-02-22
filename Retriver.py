from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load same embedding model used during indexing
model = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# Load existing vector DB
vector = Chroma(
    persist_directory="./vec_DB",
    embedding_function=model
)

# Create retriever
retriever = vector.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Query
results = retriever.invoke("men formal dress")

for doc in results:
    print(doc.page_content)
    print("------")
    