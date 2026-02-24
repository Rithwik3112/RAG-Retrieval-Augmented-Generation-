from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from Retriver import  Retriever
from langchain_core.output_parsers import StrOutputParser

class Model(Retriever):

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
        prompt = ChatPromptTemplate.from_messages([
            ("system","""Use the information from the document to answer the question. If you don't know, say you don't know. Do NOT make up an answer.
            Context:
            {context}
            """),
            ("user","Question:{question}")])
        chain = prompt | self.llm | StrOutputParser()
        for chunk in chain.stream({"context":context,"question":question}):
            yield chunk
            
if __name__ == '__main__':
    resd = Model()

    while True:
        user = input("input :")
        for res in resd.query(user):
            print(res,end="")    
        
        
        
