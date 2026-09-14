import time
import torch

from .model import Model
from .schemas import GenerateRequest, GenerateResponse
from .cache import KVCache
from .request import Request


class InferenceEngine:
    def __init__(self, model: Model):
        self.model = model
    
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
    
    # def generate_cached(self, request: Request) -> GenerateResponse:
    #     start_time = time.perf_counter()

    #     model_inputs = self.model.tokenizer(
    #         [request.prompt],
    #         return_tensors="pt",
    #     )

    #     request.input_ids = model_inputs.input_ids.to(
    #         self.model.model.device
    #     )
    #     request.input_tokens = request.input_ids.shape[1]

    #     cache = KVCache()

    #     with torch.inference_mode():

    #         # Prefill
    #         outputs = self.model.model(
    #             input_ids=request.input_ids,
    #             past_key_values=cache.get(),
    #             use_cache=True,
    #         )

    #         # Decode
    #         for _ in range(request.max_new_tokens):

    #             next_token = outputs.logits[:, -1, :].argmax(
    #                 dim=-1,
    #                 keepdim=True,
    #             )

    #             token_id = next_token.item()
    #             request.output_ids.append(token_id)

    #             # EOS → request is finished
    #             if token_id == self.model.model.config.eos_token_id:
    #                 request.finished = True
    #                 request.finish_reason = "eos"
    #                 break

    #             # Decode: process ONLY the newly generated token
    #             outputs = self.model.model(
    #                 input_ids=next_token,
    #                 past_key_values=cache.get(),
    #                 use_cache=True,
    #             )

    #         else:
    #             # Loop exhausted max_new_tokens
    #             request.finished = True
    #             request.finish_reason = "length"

    #     generated_text = self.model.tokenizer.decode(
    #         request.output_ids,
    #         skip_special_tokens=True,
    #     )

    #     latency_ms = (time.perf_counter() - start_time) * 1000

    #     return GenerateResponse(
    #         text=generated_text,
    #         input_tokens=request.input_tokens,
    #         output_tokens=request.num_generated(),
    #         latency_ms=latency_ms,
    #     )

    def prefill(self, request: Request):
        model_inputs = self.model.tokenizer(
            [request.prompt],
            return_tensors="pt",
        )

        request.input_ids = model_inputs.input_ids.to(
            self.model.model.device
        )
        request.input_tokens = request.input_ids.shape[1]

        request.cache = KVCache()

        with torch.inference_mode():
            request.outputs = self.model.model(
                input_ids=request.input_ids,
                past_key_values=request.cache.get(),
                use_cache=True,
            )

    def decode_step(self, request: Request):
        with torch.inference_mode():
            next_token = request.outputs.logits[:, -1, :].argmax(
                dim=-1,
                keepdim=True,
            )

            token_id = next_token.item()
            request.output_ids.append(token_id)

            if token_id == self.model.model.config.eos_token_id:
                request.finished = True
                request.finish_reason = "eos"
                return

            if request.num_generated() >= request.max_new_tokens:
                request.finished = True
                request.finish_reason = "length"
                return

            request.outputs = self.model.model(
                input_ids=next_token,
                past_key_values=request.cache.get(),
                use_cache=True,
            )