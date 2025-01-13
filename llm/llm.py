import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI


class BaseLLM():
    """
    OPENAI LLM Api client
    """
    model_name = None
    name = None

    def __init__(self):
        self.name = None
        self.temperature = 0.9

                
    def get_stream(self, messages):
        raise NotImplementedError
