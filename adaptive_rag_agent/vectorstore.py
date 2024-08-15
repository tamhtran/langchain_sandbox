from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class VectorStore:
    def __init__(self, urls, embedding_model=OpenAIEmbeddings(), chunk_size=500, chunk_overlap=0):
        self.urls = urls
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.docs_list = []
        self.doc_splits = []
        self.vectorstore = None

    def load_documents(self):
        docs = [WebBaseLoader(url).load() for url in self.urls]
        self.docs_list = [item for sublist in docs for item in sublist]

    def split_documents(self):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        self.doc_splits = text_splitter.split_documents(self.docs_list)

    def build_vectorstore(self, collection_name="rag-chroma"):
        self.vectorstore = Chroma.from_documents(
            documents=self.doc_splits,
            collection_name=collection_name,
            embedding=self.embedding_model,
        )
        return self.vectorstore

    def run(self):
        self.load_documents()
        self.split_documents()
        vectorstore = self.build_vectorstore()
        return vectorstore

