from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


class HallucinationGrader:
    def __init__(self, model_name="gpt-3.5-turbo-0125", temperature=0):
        # Initialize the LLM
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.structured_llm_grader = self.llm.with_structured_output(GradeHallucinations)

        # System message for grading hallucinations
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

        # Create the hallucination grading prompt
        self.hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )

        # Combine the prompt with the structured LLM grader
        self.hallucination_grader = self.hallucination_prompt | self.structured_llm_grader

    def grade_hallucination(self, documents: str, generation: str) -> str:
        # Invoke the LLM with the given documents and generation, and return the result
        return self.hallucination_grader.invoke({"documents": documents, "generation": generation})


# Data model for grading hallucinations
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# Example usage
if __name__ == "__main__":
    pass
    # # Assuming docs and generation are defined elsewhere in your code
    # hallucination_grader = HallucinationGrader()
    #
    # # Run the grader and get the result
    # result = hallucination_grader.grade_hhallucination(documents=docs, generation=generation)
    #
    # # Output the result
    # print(result)