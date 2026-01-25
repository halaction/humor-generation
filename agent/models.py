from typing import Literal

from pydantic import BaseModel, Field


class TemplateAnalysis(BaseModel):
    """Template-level interpretation of the input and joke constraints."""

    input_summary: str
    input_type: Literal["none", "topic", "headline", "constraint", "other"]
    constraint: str
    template: str
    humor_mechanics: list[str]
    tone: list[str]
    avoidances: list[str]
    reasoning_notes: list[str]


class ContextAnalysis(BaseModel):
    """Context-level reasoning grounded in the input and template."""

    topic_summary: str
    key_facts: list[str]
    cultural_context: list[str]
    comedic_angles: list[str]
    possible_targets: list[str]
    risk_notes: list[str]
    reasoning_notes: list[str]


class Association(BaseModel):
    """A narrative or association candidate used to build jokes."""

    association_id: str
    title: str
    premise: str
    connection: str
    twist: str


class AssociationSet(BaseModel):
    """A collection of narrative or association candidates."""

    associations: list[Association]


class JokeDraft(BaseModel):
    """A drafted joke tied to an association."""

    joke_id: str
    text: str
    angle: str
    rationale: str


class JokeDraftSet(BaseModel):
    """A collection of drafted jokes."""

    jokes: list[JokeDraft]


class JokeScore(BaseModel):
    """Rubric-based evaluation for a single joke."""

    joke_id: str
    readability: int = Field(ge=1, le=5)
    novelty: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    funniness: int = Field(ge=1, le=5)
    notes: str


class JokeSelection(BaseModel):
    """Judged best joke with rubric scores for all candidates."""

    best_id: str
    best_text: str
    rubric_summary: str
    scores: list[JokeScore]
    improvement_suggestions: list[str]


class RunSummary(BaseModel):
    """End-to-end artifacts captured for a single run."""

    template_analysis: TemplateAnalysis
    context_analysis: ContextAnalysis
    associations: AssociationSet
    drafts: JokeDraftSet
    selection: JokeSelection
