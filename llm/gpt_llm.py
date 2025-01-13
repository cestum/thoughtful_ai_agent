import time

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI, OpenAIError
import tiktoken

from .llm import BaseLLM


class GPTLLM(BaseLLM):
    """
    GPT LLM Api client
    """

    def __init__(self, *args, **kwargs):
        self.temperature = kwargs.get("GPT_TEMPERATURE",0.1)
        self.max_tokens = kwargs.get("GPT_MAX_TOKENS", 2048)
        self.encoding_model = kwargs.get("GPT_ENCODING_MODEL", "o200k_base")
        self.API_MAX_RETRY = kwargs.get("GPT_API_MAX_RETRY", 5)
        self.model_name = kwargs.get("GPT_MODEL_NAME", "gpt-4o")
        # self.llm = ChatOpenAI(self.temperature, max_tokens=2048)
        self.client = OpenAI()

    def truncate_user_message(self, message, suffix="..."):
        enc = tiktoken.get_encoding(self.encoding_model)
        tokens = enc.encode(message)
        if len(tokens) > self.max_tokens:
            truncated_token = token[:self.max_tokens]
            truncated_message = enc.decode(truncated_token)
            return truncated_message + suffix
        else:
            return message + suffix


    def get_stream(self, messages):
        messages[-1]["content"] = self.truncate_user_message(message=messages[-1]["content"])
        output = ""
        for _ in range(self.API_MAX_RETRY):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name, 
                    messages=messages, 
                    stream=True,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=1.0
                )
                # output = response.choices[0].message.content
                for chunk in response:
                    yield chunk.choices[0].delta.content
                break
            except OpenAIError:
                print("OpenAI API error, retrying after 10 sec")
                time.sleep(10)
                output = "There was an error processing your question. We have notified admin. Try again later."
            
        yield output
