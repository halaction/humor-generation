from pydantic import BaseModel


class JokesItem(BaseModel):
    id: str
    text: str


class EmbeddingsInputs(BaseModel):
    id: list[str]
    text: list[str]


class EmbeddingsItem(BaseModel):
    id: str
    embedding: list[float]


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
    id: str
    keywords: list[str]


class ReferencesOutputs(BaseModel):
    id: str
    prompt: str
    references: list[str]
    scores: list[float]
