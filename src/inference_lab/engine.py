import time

from .model import Model
from .schemas import GenerateRequest, GenerateResponse


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
        generated_ids = self.model.model.generate(**model_inputs, max_new_tokens=max_new_tokens)
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