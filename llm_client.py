from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI

from singleton import Singleton

class LLMClient(metaclass=Singleton):
    """
    simple LLM Api client
    """
    model_name = "gpt-4o"

    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.1, max_tokens=2048) #not so creative
        self.client = OpenAI()
        #We will add embeddings for vector db embedding
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    def get_stream(self, messages):
        return self.client.chat.completions.create(model=self.model_name, messages=messages, stream=True)

