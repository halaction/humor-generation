from pydantic import BaseModel
from pathlib import Path


class JokesItem(BaseModel):
    id: str
    text: str


class JokesBatch(BaseModel):
    id: list[str]
    text: list[str]


class EmbeddingsItem(BaseModel):
    id: str
    embedding: list[float]


class EmbeddingsBatch(BaseModel):
    id: list[str]
    embedding: list[list[float]]


class EmbeddingsOutputs(BaseModel):
    output_path: Path


class KeywordsItem(BaseModel):
    joke_id: str
    keywords: list[str]
    scores: list[float]


class KeywordsOutputs(BaseModel):
    output_path: Path
