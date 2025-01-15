import os

from transformers import (
    pipeline, 
    AutoModelForCausalLM, 
    AutoModel
)
import torch
from .models import MODEL_PATHS

class TFLLM():
    """
    Transformer Api client
    """
    model_name = None
    name = None

    def __init__(self, model):
        self.name = None
        self.temperature = 0.9
        self.max_tokens = 2048
        self.top_p=0.9
        self.model_path = MODEL_PATHS.get(model)
        self.pipeline = pipeline(
            "text-generation",
            model=self.model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"
        )

                
    def get_stream(self, messages):
        outputs = self.pipeline(
            messages,
            max_new_tokens=self.max_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,            
        )
        return outputs[0]["generated_text"][-1]['content'].strip()
        