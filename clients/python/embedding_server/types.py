from enum import Enum
from pydantic import BaseModel, validator
from typing import Optional, List

from embedding_server.errors import ValidationError


class Request(BaseModel):
    # Prompt
    inputs: str

    @validator("inputs")
    def valid_input(cls, v):
        if not v:
            raise ValidationError("`inputs` cannot be empty")
        return v


# `generate` return value
class Response(BaseModel):
    # embedding for the passed input
    embedding: List[float]
    # dimensions of the embedding
    dim: int


# Inference API currently deployed model
class DeployedModel(BaseModel):
    model_id: str
    sha: str
