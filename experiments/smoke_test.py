from inference_lab.model import Model
from inference_lab.engine import InferenceEngine
from inference_lab.schemas import GenerateRequest

model = Model(
    model_name="sshleifer/tiny-gpt2",
    device="cpu",
)

engine = InferenceEngine(model)

request = GenerateRequest(
    prompt="Explain gravity",
    max_new_tokens=50,
)

response = engine.generate(request)

print(response.text)