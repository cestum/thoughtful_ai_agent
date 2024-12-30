import json

from langchain.storage import InMemoryByteStore
from langchain.retrievers import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_chroma import Chroma

from singleton import Singleton
from abstract_rag_retriever import AbstractRAGRetriever

class MVRetreiver(AbstractRAGRetriever):
    """
    Class to build RAG database. It uses chroma vector DB. 
    We will use Questions and ANswers vectors in db. 
    MultiVectorRetriever could be a good use here.
    May be start with Questions and add answers to metadata.
    """

    def __init__(self,  data, llm_client):
        self.vectorstore = Chroma(
            collection_name="thoughtfulai", 
            persist_directory="db", 
            embedding_function=llm_client.embeddings
        )
        self.memstore = InMemoryByteStore()
        self.retriever = self.get_rag_retriever(data)

    def get_rag_retriever(self, data_location):
        docs = {}
        with open(data_location, "r") as data_reader:
            docs = json.load(data_reader)
        
        qas = [] #question document
        qas_ids = [] #for document store 
        for idx, doc in enumerate(docs["questions"]):
            #question as primary vector and add answer to metadata
            qas.append(
                Document(
                    page_content=doc["question"], 
                    metadata={
                        "doc_id":idx, 
                        "answer": doc["answer"]
                    }
                )
            )
            qas_ids.append(idx)
        
        #considering multivector with questions and answers as separate vector
        retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore, 
            byte_store=self.memstore, 
            id_key="doc_id"
        )
        retriever.vectorstore.add_documents(qas)
        retriever.docstore.mset(list(zip(qas_ids,qas))) #for now set id and question vector
        return retriever

    def query(self, query, result_limit=1):
        results= self.retriever.vectorstore.max_marginal_relevance_search(query, k=result_limit)
        out = []
        for result in results:
            out.append({
                "metadata": result.metadata,
                "content": result.page_content,
            })        
        return out