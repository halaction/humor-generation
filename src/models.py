from typing import Literal
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
    keywords: list[list[str]]
    references: list[list[str]]
    scores: list[list[float]]


class CandidateOutput(BaseModel):
    id: int
    keywords: list[str]
    model: str
    text: str


class EvaluationCandidate(BaseModel):
    id: int
    keywords: list[str]
    model: str
    text: str


class EvaluationPair(BaseModel):
    id: int
    reference_id: int
    prompt: str
    left_model: str
    right_model: str
    left_text: str
    right_text: str


class EvaluationJudgeDecision(BaseModel):
    winner: Literal["left", "right"]


class EvaluationOutputs(BaseModel):
    id: list[int]
    reference_id: list[int]
    prompt: list[str]
    left_model: list[str]
    right_model: list[str]
    left_text: list[str]
    right_text: list[str]
    winner: list[str]
