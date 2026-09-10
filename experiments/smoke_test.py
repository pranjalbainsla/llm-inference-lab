from inference_lab.model import Model
from inference_lab.engine import InferenceEngine
from inference_lab.schemas import GenerateRequest
from inference_lab.config import Settings

settings = Settings()

model = Model(
    model_name=settings.model_name,
    device=settings.device,
)

engine = InferenceEngine(model)

request = GenerateRequest(
    prompt="Explain gravity",
    max_new_tokens=50,
)

response = engine.generate(request)

print(response.text)