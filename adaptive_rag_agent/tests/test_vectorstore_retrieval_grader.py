import unittest
from vectorstore import VectorStore
from retrieval_grader_agent import RetrieverGrader

class MyTestCase(unittest.TestCase):
    def test_vectorstore_retrival_grader(self):
        # Example usage
        urls = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ]

        retriever = VectorStore(urls=urls).run().as_retriever()

        retriever_grader = RetrieverGrader()

        # Assuming `retriever` is an instance of a retriever that can invoke a search
        question = "agent memory"
        docs = retriever.invoke(question)

        if docs and len(docs) > 1:
            doc_txt = docs[1].page_content
        else:
            doc_txt = "No document retrieved."

        result = retriever_grader.grade_document(question, doc_txt)
        assert result.binary_score == "yes"


if __name__ == '__main__':
    unittest.main()
