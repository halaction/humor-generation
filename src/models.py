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


class EvaluationCandidate(BaseModel):
    id: int
    prompt_id: str
    prompt: str
    model_id: str
    model: str
    text: str


class EvaluationPair(BaseModel):
    id: int
    prompt_id: str
    prompt: str
    left_candidate_id: int
    right_candidate_id: int
    left_model_id: str
    right_model_id: str
    left_model: str
    right_model: str
    left_text: str
    right_text: str


class EvaluationJudgeDecision(BaseModel):
    winner: Literal["left", "right"]


class EvaluationOutputs(BaseModel):
    id: list[int]
    prompt_id: list[str]
    prompt: list[str]
    left_candidate_id: list[int]
    right_candidate_id: list[int]
    left_model_id: list[str]
    right_model_id: list[str]
    left_model: list[str]
    right_model: list[str]
    left_text: list[str]
    right_text: list[str]
    winner: list[str]
