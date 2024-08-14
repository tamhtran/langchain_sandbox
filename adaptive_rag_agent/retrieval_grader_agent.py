from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


# Data model for grading documents
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class RetrieverGrader:
    def __init__(self, model_name="gpt-3.5-turbo-0125", temperature=0):
        # Initialize the LLM
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        system = """You are a grader assessing relevance of a retrieved document to a user question. 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

        # Create and return the prompt template
        self.grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )

        # Combine the prompt with the structured LLM grader
        self.retrieval_grader = self.grade_prompt | self.structured_llm_grader


    def grade_document(self, question: str, document: str) -> str:
        # Invoke the LLM with the given question and document, and return the result
        return self.retrieval_grader.invoke( {"question": question, "document": document} )



# Example usage
if __name__ == "__main__":
    retriever_grader = RetrieverGrader()

    # Assuming `retriever` is an instance of a retriever that can invoke a search
    question = "agent memory"
    docs = retriever.invoke(question)

    if docs and len(docs) > 1:
        doc_txt = docs[1].page_content
    else:
        doc_txt = "No document retrieved."

    result = retriever_grader.grade_document(question, doc_txt)
    print(result)