import time

from inference_lab.engine import InferenceEngine
from inference_lab.model import Model
from inference_lab.schemas import GenerateRequest

model = Model(
    model_name="sshleifer/tiny-gpt2",
    device="cuda",
)

engine = InferenceEngine(model)

for n in [10, 50, 100, 200]:
    request = GenerateRequest(
        prompt="what is quantization",
        max_new_tokens=n,
    )

    naive = engine.generate_naive(request)
    cached = engine.generate_cached(request)

    print(f"\n{n} new tokens")
    print(f"naive:  {naive.latency_ms:.2f} ms")
    print(f"cached: {cached.latency_ms:.2f} ms")
    print(f"naive:  {n / (naive.latency_ms / 1000):.2f} tok/s")
    print(f"cached: {n / (cached.latency_ms / 1000):.2f} tok/s")