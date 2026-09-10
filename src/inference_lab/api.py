from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging
from .model import Model
from .config import Settings
from .engine import InferenceEngine
from .schemas import GenerateRequest, GenerateResponse

logger = logging.getLogger(__name__)

settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    logger.info("Loading model...")
    model = Model(
        model_name=settings.model_name,
        device=settings.device,
    )
    engine = InferenceEngine(model)
    app.state.engine = engine

    yield

    # shutdown
    logger.info("Cleaning up...")
    del app.state.engine

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/generate")
def generate(request: GenerateRequest) -> GenerateResponse:
    return app.state.engine.generate(request)