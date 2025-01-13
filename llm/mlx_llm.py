import os

from mlx_lm import stream_generate, load
from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache

from .models import MODEL_PATHS

class MLXLLM():
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
        self.model, self.tokenizer = load(
            self.model_path,
            tokenizer_config={"trust_remote_code": True},
        )
        self.prompt_cache = make_prompt_cache(self.model)

                
    def get_stream(self, messages):
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        for resp in stream_generate(
            self.model,
            self.tokenizer,
            prompt=input_text,
            max_tokens=self.max_tokens,
            prompt_cache=self.prompt_cache
        ):
            yield resp.text

        