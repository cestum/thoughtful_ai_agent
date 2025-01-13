class AbstractRAGRetriever:
    def get_rag_retriever(self):
        raise NotImplementedError
    
    def query(self, query, result_limit=1):
        raise NotImplementedError