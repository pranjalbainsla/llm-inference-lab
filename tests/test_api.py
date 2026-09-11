from fastapi.testclient import TestClient
from inference_lab.api import create_app
from inference_lab.schemas import GenerateResponse

class FakeEngine():
    def generate(self, request):
        return GenerateResponse(
            text="fake response",
            input_tokens=3,
            output_tokens=2,
            latency_ms=1.0,
        )

def test_generate():
    fake_engine = FakeEngine()
    app = create_app(engine=fake_engine)

    with TestClient(app) as client:
        response = client.post(
            "/generate",
            json={"prompt": "hello", "max_new_tokens": 2},
        )

    assert response.status_code == 200
    assert response.json() == {
        "text": "fake response",
        "input_tokens": 3,
        "output_tokens": 2,
        "latency_ms": 1.0,
    }