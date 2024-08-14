from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


class QuestionRewriter:
    def __init__(self, model_name="gpt-3.5-turbo-0125", temperature=0):
        # Initialize the LLM
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.output_parser = StrOutputParser()

        # System message for rewriting the question
        system = """You are a question re-writer that converts an input question to a better version that is optimized 
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""

        # Create the question rewriting prompt
        self.rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                ),
            ]
        )

        # Combine the prompt with the LLM and output parser
        self.question_rewriter = self.rewrite_prompt | self.llm | self.output_parser

    def rewrite_question(self, question: str) -> str:
        # Invoke the LLM to rewrite the question and return the result
        return self.question_rewriter.invoke({"question": question})


# Example usage
if __name__ == "__main__":
    question = "What is the significance of quantum entanglement?"  # Replace with your question
    # Assuming question is defined elsewhere in your code
    question_rewriter = QuestionRewriter()

    # Run the rewriter and get the result
    result = question_rewriter.rewrite_question(question=question)

    # Output the result
    print(result)