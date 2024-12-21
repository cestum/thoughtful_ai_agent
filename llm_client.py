from langchain_openai import ChatOpenAI
from openai import OpenAI

class LLMClient():
    """
    simple LLM Api client
    """
    model_name = "gpt-4o"

    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.1, max_tokens=2048) #not so creative
        self.client = OpenAI()

    def get_stream(self, messages):
        return self.client.chat.completions.create(model=self.model_name, messages=messages, stream=True)

