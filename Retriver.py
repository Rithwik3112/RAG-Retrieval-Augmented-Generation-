from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class Retriever:
    def __init__(self):
        
        model = HuggingFaceEmbeddings(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        vector = Chroma(
            persist_directory="./vec_DB",
            embedding_function=model
        )

        self.retriever = vector.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

    def ret(self, query:str):
        results = self.retriever.invoke(query)

        result = ""
        for doc in results:
            result += doc.page_content + "\n------\n"

        return result
if __name__ == '__main__':
    res = Retriever()
    while True:
        user = input("input :")
        print(res.ret(user))
