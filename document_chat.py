from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from utils import load_embeddings, load_db
# from dotenv import load_dotenv
OPENAI_API_KEY = "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"
# Load environment variables from .env file
# load_dotenv()
class retrieval_chat():

    def __init__(self) -> None:
        
        embedding_function = load_embeddings()

        db = load_db(embedding_function)

        self.qa = RetrievalQA.from_llm(
            llm=ChatOpenAI(temperature=0.1, openai_api_key="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"),
            retriever=db.as_retriever(kwargs={"k": 7}),
            return_source_documents=True
)


    def answer_question(self, question :str):
        output = self.qa({"query": question})
        print("Source Documents: " + ", ".join(map(str, output["source_documents"])))
        return output["result"]

if __name__ == "__main__":
    qa = retrieval_chat()
    while True:
        print("Whats Your Question:")
        query = input()
        if query == "exit":
            break
        print(qa.answer_question(query))
