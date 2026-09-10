from pydantic import BaseModel, Field

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    max_new_tokens: int = Field(default=100, ge=1, le=1000)

class GenerateResponse(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float