from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_name: str = "sshleifer/tiny-gpt2"
    device: str = "cuda"