from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.runnables import RunnableLambda
from Retriver import  Retriever

class model(Retriever):

    def __init__(self):
        super().__init__()
        model =HuggingFaceEndpoint(
            model="meta-llama/Llama-3.2-1B-Instruct",
            task="text-generation",
            max_new_tokens=2000,
            do_sample=False,
            repetition_penalty=1.03,
            provider="auto",  # let Hugging Face choose the best provider for you
        )
        self.llm = ChatHuggingFace(llm = model)
    def query(self,question):
        context = self.ret(question)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system","""Use the information from the document to answer the question. If you don't know, say you don't know. Do NOT make up an answer.
            Context:
            {context}
            """),
            ("user","Question:{question}")])
        return self.prompt
if __name__ == '__main__':
    resd = model()
    while True:
        user = input("input :")
        print(resd.query(user))
    
        
        
        
