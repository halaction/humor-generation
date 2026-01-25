from dataclasses import dataclass

from openai import OpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from agent.models import (
    AssociationSet,
    ContextAnalysis,
    JokeDraftSet,
    JokeSelection,
    TemplateAnalysis,
)
from agent.prompts import (
    ASSOCIATION_SYSTEM_PROMPT,
    CONTEXT_SYSTEM_PROMPT,
    DRAFT_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    TEMPLATE_SYSTEM_PROMPT,
)
from agent.settings import settings


@dataclass(frozen=True)
class Agents:
    template: Agent
    context: Agent
    associations: Agent
    drafts: Agent
    judge: Agent


def build_openai_client() -> OpenAI:
    return OpenAI(api_key=settings.api_key, base_url=settings.base_url)


def build_openai_model(model_name: str) -> OpenAIChatModel:
    provider = OpenAIProvider(api_key=settings.api_key, base_url=settings.base_url)
    return OpenAIChatModel(model_name, provider=provider)


def get_agents(model_name: str) -> Agents:
    client = build_openai_client()
    model = build_openai_model(model_name)

    template_agent = Agent(
        model=model,
        output_type=TemplateAnalysis,
        system_prompt=TEMPLATE_SYSTEM_PROMPT,
    )
    context_agent = Agent(
        model=model,
        output_type=ContextAnalysis,
        system_prompt=CONTEXT_SYSTEM_PROMPT,
    )
    association_agent = Agent(
        model=model,
        output_type=AssociationSet,
        system_prompt=ASSOCIATION_SYSTEM_PROMPT,
    )
    draft_agent = Agent(
        model=model,
        output_type=JokeDraftSet,
        system_prompt=DRAFT_SYSTEM_PROMPT,
    )
    judge_agent = Agent(
        model=model,
        output_type=JokeSelection,
        system_prompt=JUDGE_SYSTEM_PROMPT,
    )

    return Agents(
        template=template_agent,
        context=context_agent,
        associations=association_agent,
        drafts=draft_agent,
        judge=judge_agent,
    )
