from langsmith import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


class RAGAgent:
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0, prompt_id="rlm/rag-prompt"):
        # Initialize the client and LLM
        self.client = Client()
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

        # Pull the prompt using the client
        self.prompt = self.client.pull_prompt(prompt_id)

        # Initialize the output parser
        self.output_parser = StrOutputParser()

        # Combine the prompt with the LLM and output parser into a chain
        self.rag_chain = self.prompt | self.llm | self.output_parser

    @staticmethod
    def format_docs(docs):
        # Format the documents for input into the chain
        return "\n\n".join(doc.page_content for doc in docs)

    def run_chain(self, docs, question: str) -> str:
        # Format the documents
        formatted_docs = self.format_docs(docs)

        # Invoke the chain with the formatted documents and question
        return self.rag_chain.invoke({"context": formatted_docs, "question": question})

