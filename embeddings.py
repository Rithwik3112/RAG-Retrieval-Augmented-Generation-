from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


loader = PyPDFLoader("data.pdf")
data = loader.load()

text_sp = RecursiveCharacterTextSplitter(
    chunk_size= 500,
    chunk_overlap = 50
)
doc = text_sp.split_documents(data)
model = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
vector = Chroma.from_documents(
    documents=doc,
    embedding=model,
    persist_directory="./vec_DB"
)
vector.persist()
print("PDF converted to vector embeddings successfully!")
