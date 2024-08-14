from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


class AnswerGrader:
    def __init__(self, model_name="gpt-3.5-turbo-0125", temperature=0):
        # Initialize the LLM
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.structured_llm_grader = self.llm.with_structured_output(GradeAnswer)

        # System message for grading the answer
        system = """You are a grader assessing whether an answer addresses / resolves a question. 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""

        # Create the grading prompt
        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )

        # Combine the prompt with the structured LLM grader
        self.answer_grader = self.answer_prompt | self.structured_llm_grader

    def grade_answer(self, question: str, generation: str) -> str:
        # Invoke the LLM with the given question and generation, and return the result
        return self.answer_grader.invoke({"question": question, "generation": generation})


# Data model for grading answers
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# Example usage
if __name__ == "__main__":
    pass
    # # Assuming question and generation are defined elsewhere in your code
    # answer_grader = AnswerGrader()
    #
    # # Run the grader and get the result
    # result = answer_grader.grade_answer(question=question, generation=generation)
    #
    # # Output the result
    # print(result)