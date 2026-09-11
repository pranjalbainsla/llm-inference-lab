from fastapi.testclient import TestClient
from inference_lab.api import create_app
from inference_lab.schemas import GenerateResponse


class FakeEngine:
    def generate(self, request):
        return GenerateResponse(
            text="fake response",
            input_tokens=3,
            output_tokens=2,
            latency_ms=1.0,
        )


def test_health():
    app = create_app(FakeEngine())

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_generate():
    app = create_app(FakeEngine())

    with TestClient(app) as client:
        response = client.post(
            "/generate",
            json={
                "prompt": "explain deep learning",
                "max_new_tokens": 2,
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "text": "fake response",
        "input_tokens": 3,
        "output_tokens": 2,
        "latency_ms": 1.0,
    }


def test_generate_rejects_empty_prompt():
    app = create_app(FakeEngine())

    with TestClient(app) as client:
        response = client.post(
            "/generate",
            json={
                "prompt": "",
                "max_new_tokens": 2,
            },
        )

    assert response.status_code == 422


def test_generate_rejects_invalid_max_new_tokens():
    app = create_app(FakeEngine())

    with TestClient(app) as client:
        response = client.post(
            "/generate",
            json={
                "prompt": "hello",
                "max_new_tokens": 0,
            },
        )

    assert response.status_code == 422