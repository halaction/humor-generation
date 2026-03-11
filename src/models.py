from pydantic import BaseModel


class JokesItem(BaseModel):
    id: int
    text: str


class EmbeddingsInputs(BaseModel):
    id: list[int]
    text: list[str]


class EmbeddingsOutputs(BaseModel):
    id: list[int]
    embedding: list[list[float]]


class KeywordsInputs(BaseModel):
    id: int
    text: str
    embedding: list[float]


class KeywordsOutputs(BaseModel):
    id: int
    keywords: list[str]
    scores: list[float]


class ReferencesInputs(BaseModel):
    id: list[int]
    keywords: list[list[str]]


class ReferencesOutputs(BaseModel):
    id: list[int]
    prompt: list[str]
    references: list[list[str]]
    scores: list[list[float]]
