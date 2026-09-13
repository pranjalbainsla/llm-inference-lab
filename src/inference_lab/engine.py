import time
import torch

from .model import Model
from .schemas import GenerateRequest, GenerateResponse
from .cache import KVCache


class InferenceEngine:
    def __init__(self, model: Model):
        self.model = model
    
    def generate_naive(self, request: GenerateRequest) -> GenerateResponse:
        prompt, max_new_tokens = request.prompt, request.max_new_tokens

        start_time = time.perf_counter()

        model_inputs = self.model.tokenizer(
            [prompt],
            return_tensors="pt",
        )

        input_ids = model_inputs.input_ids.to(self.model.model.device)
        input_tokens = input_ids.shape[1]

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                outputs = self.model.model(input_ids=input_ids)

                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True) # (B, 1)

                input_ids = torch.cat([input_ids, next_token], dim=1) # (B, T + max_new_tokens)

        generated_text = self.model.tokenizer.batch_decode(
            input_ids[:, input_tokens:],
            skip_special_tokens=True,
        )[0]

        return GenerateResponse(
            text=generated_text,
            input_tokens=input_tokens,
            output_tokens=input_ids.shape[1] - input_tokens,
            latency_ms=(time.perf_counter() - start_time) * 1000,
        )

    def generate_cached(self, request: GenerateRequest) -> GenerateResponse:
        prompt, max_new_tokens = request.prompt, request.max_new_tokens

        start_time = time.perf_counter()

        model_inputs = self.model.tokenizer(
            [prompt],
            return_tensors="pt",
        )

        input_ids = model_inputs.input_ids.to(self.model.model.device)
        input_tokens = input_ids.shape[1]

        cache = KVCache()

        with torch.inference_mode():
            # Prefill: process the entire prompt once
            outputs = self.model.model(
                input_ids=input_ids,
                past_key_values=cache.get(),
                use_cache=True,
            )

            for _ in range(max_new_tokens):
                next_token = outputs.logits[:, -1, :].argmax(
                    dim=-1,
                    keepdim=True,
                )

                # Decode: process ONLY the new token
                outputs = self.model.model(
                    input_ids=next_token,
                    past_key_values=cache.get(),
                    use_cache=True,
                )

                input_ids = torch.cat([input_ids, next_token], dim=1)

        generated_text = self.model.tokenizer.batch_decode(
            input_ids[:, input_tokens:],
            skip_special_tokens=True,
        )[0]
        
        latency_ms = (time.perf_counter() - start_time) * 1000

        return GenerateResponse(
            text=generated_text,
            input_tokens=input_tokens,
            output_tokens=input_ids.shape[1] - input_tokens,
            latency_ms=latency_ms,
        )

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        prompt, max_new_tokens = request.prompt, request.max_new_tokens

        start_time = time.perf_counter()
        
        model_inputs = self.model.tokenizer(
            [prompt], 
            return_tensors="pt"
        ) 
        input_tokens = model_inputs.input_ids.shape[1]
        model_inputs = {
            key: value.to(self.model.model.device)
            for key, value in model_inputs.items()
        }
        
        with torch.inference_mode():
            generated_ids = self.model.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
            )
        
        generated_text = self.model.tokenizer.batch_decode(
            generated_ids[:, input_tokens:], 
            skip_special_tokens=True
        )[0]
        
        return GenerateResponse(
            text=generated_text,
            input_tokens=input_tokens,
            output_tokens=generated_ids.shape[1]-input_tokens,
            latency_ms=(time.perf_counter() - start_time) * 1000,
        )