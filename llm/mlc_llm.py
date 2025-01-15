import os
import torch
import logging
from mlc_llm import MLCEngine

from .models import MODEL_PATHS

logger = logging.getLogger(__name__)

class MLCLLM():
    """
    MLC-LLM client
    """
    model_name = None
    name = None

    def __init__(self, model):
        self.model_path = MODEL_PATHS.get(model)
        logger.info("Using %s model in MLCLLM", self.model_path )
        self.engine = MLCEngine(self.model_path)
    
    def get_stream(self, messages):
        for response in self.engine.chat.completions.create(
            messages=messages,
            model=self.model_path,
            stream=True,
        ):
            for choice in response.choices:
                yield choice.delta.content
