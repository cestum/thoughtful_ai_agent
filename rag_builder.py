import json
from langchain_core.documents import Document
from langchain_chroma import Chroma

class RAGBuilder:
    """
    Class to build RAG database. It uses chroma vector DB. 
    We will use Questions and ANswers vectors in db. 
    MultiVectorRetriever could be a good use here.
    May be start with Questions and add answers to metadata.
    """

    def __init__(self, llm_client, data_location):
        docs = {}
        with open(data_location, "r") as data_reader:
            docs = json.load(data_reader)
        
        qas = [] #question document
        for idx, doc in enumerate(docs["questions"]):
            qas.append(Document(page_content=doc["question"], metadata={"doc_id":idx, "answer": doc["answer"]}))
        
        vectorstore = Chroma(
            collection_name="thoughtfulai", 
            persist_directory="db", 
            embedding_function=llm_client.embeddings
        )