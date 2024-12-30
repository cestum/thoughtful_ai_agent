import json

from haystack import Document
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder
)
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


from singleton import Singleton
from abstract_rag_retriever import AbstractRAGRetriever

class InmemoryRetreiver(AbstractRAGRetriever):
    """
    Class to build RAG database. It uses InMemoryDocumentStore. 
    """

    def __init__(self,  data, llm_client):
        self.text_embedder = SentenceTransformersTextEmbedder()
        self.text_embedder.warm_up()
        self.memstore = InMemoryDocumentStore()
        self.retriever = self.get_rag_retriever(data)

    def get_rag_retriever(self, data_location):
        docs = {}
        with open(data_location, "r") as data_reader:
            docs = json.load(data_reader)
        
        documents = [   
            Document(content=qa["question"], meta={"answer": qa["answer"]})
            for qa in docs["questions"]
        ]
        doc_retriever = SentenceTransformersDocumentEmbedder()
        doc_retriever.warm_up()
        doc_embeddings = doc_retriever.run(documents)["documents"]
        self.memstore.write_documents(doc_embeddings)
        return InMemoryEmbeddingRetriever(self.memstore)

    def query(self, query, result_limit=1):
        _embedded_query = self.text_embedder.run(query)["embedding"]
        results = self.retriever.run(query_embedding=_embedded_query, top_k=result_limit)["documents"]
        out = []
        for result in results:
            out.append({
                "metadata": result.meta,
                "content": result.content,
                "score": result.score
            })
        return out
