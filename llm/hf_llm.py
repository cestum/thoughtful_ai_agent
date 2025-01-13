import os

import torch
from transformers import (
    pipeline, 
    AutoModelForCausalLM, 
    AutoTokenizer
)

from .models import MODEL_PATHS

class HuggingfaceLLM():
    """
    Huggingface Api client
    """
    model_name = None
    name = None

    def __init__(self, model):
        self.name = None
        self.temperature = 0.9
        self.max_tokens = 2048
        self.top_p=0.9
        self.model_path = MODEL_PATHS.get(model)
        if token := os.getenv("HF_TOKEN"):
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.model_path,
                trust_remote_code=True
            )
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            ).eval()
        else:
            raise Exception("Set HUGGING_FACE_TOKEN API Token")
        self.has_chat_template = hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None

    def gen_with_template(self, messages, max_input_tokens):
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)
        # input_torch = self.tokenizer([input_text], return_tensors="pt")
        generated_tokens = self.model.generate(
            **input_text,
            max_new_tokens=max_input_tokens,
            temperature=self.temperature,
            do_sample=True
        )

        if not self.model.config.is_encoder_decoder: #strip off input tokens
            generated_tokens = [   
                output_tokens[len(inputids):]
                for output_tokens, inputids in zip(generated_tokens, inputs["input_ids"])
            ]
        outputs_list = self.tokenizer.batch_decode(
            generated_tokens, 
            skip_special_tokens=True
        )[0]
        assistant_response = outputs_list.split("assistant:")[-1].strip()        
        return assistant_response        
        

    def gen_without_template(self, messages, max_input_tokens):
        # Combine messages into a single string
        # Tokenize the input
        inputs = self.tokenizer(messages, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)

        # Generate
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_input_tokens, temperature=0.7, do_sample=True)

        # Decode and return the generated text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        assistant_response = generated_text.split("assistant:")[-1].strip()        
        return assistant_response
                
    def get_stream(self, messages):
        max_input_tokens = min(
            self.model.config.max_position_embeddings, self.max_tokens
        )
        # inputs = self.tokenizer.apply_chat_template(
        #     messages,
        #     return_tensors="pt",
        #     truncation=True,
        #     padding=True,
        #     max_length=max_input_tokens,
        #     return_dict=True
        # )
        # input_torch = inputs.to(self.model.device)

        if self.has_chat_template:
            assistant_response = self.gen_with_template(messages,max_input_tokens)
        else:
            assistant_response = self.gen_without_template(messages,max_input_tokens)

        yield assistant_response
        # for output in outputs_list:
        #     yield output.strip()
