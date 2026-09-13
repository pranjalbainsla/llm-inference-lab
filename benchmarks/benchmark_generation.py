import torch
import time
import statistics
from inference_lab.engine import InferenceEngine
from inference_lab.model import Model
from inference_lab.schemas import GenerateRequest

def benchmark(fn, request, repetitions=5):
    times = []

    for _ in range(repetitions):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        response = fn(request)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    median_ms = statistics.median(times)

    return {
        "latency_ms": median_ms,
        "tokens_per_sec": request.max_new_tokens / (median_ms / 1000),
    }

model = Model(
    model_name="sshleifer/tiny-gpt2",
    device="cuda",
)

engine = InferenceEngine(model)

results = []

for n in [10, 50, 100, 200]:
    request = GenerateRequest(
        prompt="The future of artificial intelligence is",
        max_new_tokens=n,
    )

    # Warmup
    for _ in range(3):
        engine.generate_naive(request)
        engine.generate_cached(request)

    naive = benchmark(engine.generate_naive, request)
    cached = benchmark(engine.generate_cached, request)

    results.append({
        "tokens": n,
        "naive_ms": naive["latency_ms"],
        "cached_ms": cached["latency_ms"],
        "naive_tok_s": naive["tokens_per_sec"],
        "cached_tok_s": cached["tokens_per_sec"],
    })
print(results)