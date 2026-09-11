import torch

from inference_lab.engine import InferenceEngine
from inference_lab.schemas import GenerateRequest


class FakeTokenizer:
    class ModelInputs(dict):
        @property
        def input_ids(self):
            return self["input_ids"]

    def __call__(self, prompts, return_tensors):
        return self.ModelInputs(
            input_ids=torch.tensor([[0, 0, 0, 0, 0]]),
            attention_mask=torch.tensor([[1, 1, 1, 1, 1]]),
        )

    def batch_decode(self, token_ids, skip_special_tokens):
        return ["fake output"]

class FakeModel:
    def __init__(self):
        self.device = torch.device("cpu")

    def generate(self, **model_inputs):
        return torch.tensor([
            [0, 0, 0, 0, 0, 1, 1, 1]
        ])


class FakeModelWrapper:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.model = FakeModel()


def test_generate():
    model = FakeModelWrapper()
    engine = InferenceEngine(model)

    request = GenerateRequest(
        prompt="hello",
        max_new_tokens=3,
    )

    response = engine.generate(request)

    assert response.text == "fake output"
    assert response.input_tokens == 5
    assert response.output_tokens == 3