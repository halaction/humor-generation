from pydantic import BaseModel


class JokesItem(BaseModel):
    id: str
    text: str


class EmbeddingsInputs(BaseModel):
    id: list[str]
    text: list[str]


class EmbeddingsOutputs(BaseModel):
    id: list[str]
    embedding: list[list[float]]


class KeywordsInputs(BaseModel):
    id: str
    text: str
    embedding: list[float]


class KeywordsOutputs(BaseModel):
    id: str
    keywords: list[str]
    scores: list[float]


class ReferencesInputs(BaseModel):
    id: list[str]
    joke: list[str]
    keywords: list[list[str]]


class ReferencesOutputs(BaseModel):
    id: list[str]
    prompt: list[str]
    references: list[list[str]]
    scores: list[list[float]]
