from pydantic.v1 import BaseModel, Field, validator
from typing import Any, Optional

class EmbeddingConfig(BaseModel):
    name: str = Field(..., description="The name of the SentenceTransformer model")
    device: str = Field('cpu', description="Device to run embedding model: cpu | cuda | mps")
    @validator('name', allow_reuse=True)
    def check_model_name(cls, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Model name must be a non-empty string")
        return value
    @validator("device", allow_reuse=True)
    def check_device(cls, value):
        allowed = {"cpu", "cuda", "mps"}
        value = value.lower()
        if value not in allowed:
            raise ValueError(
                f"Device must be one of {allowed}, got '{value}'"
            )
        return value
    
class APIEmbeddingConfig(BaseModel):
    name: str = Field(..., description="The name of the API model")
    apiKey: str = Field(..., description="The api key cor verification")

    @validator('name', allow_reuse=True)
    def check_model_name(cls, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Model name must be a non-empty string")
        return value
    
    @validator('apiKey', allow_reuse=True)
    def check_api_key(cls, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("API key must be a non-empty string")
        return value


class BaseEmbedding():
    name: str

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def encode(self, text: str):
        raise NotImplementedError("The encode method must be implemented by subclasses")


class APIBaseEmbedding(BaseEmbedding):
    baseUrl: str
    apiKey: str

    def __init__(self, name: str = None, baseUrl: str = None, apiKey: str = None):
        super().__init__(name)
        self.baseUrl = baseUrl
        self.apiKey = apiKey