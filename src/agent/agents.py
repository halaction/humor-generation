from dataclasses import dataclass

from openai import OpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from agent.prompts import (
    ANGLES_SYSTEM_PROMPT,
    CONTEXT_SYSTEM_PROMPT,
    DRAFTS_SYSTEM_PROMPT,
    SELECTION_SYSTEM_PROMPT,
    TEMPLATE_SYSTEM_PROMPT,
)
from agent.settings import settings


@dataclass(frozen=True)
class Agents:
    template: Agent
    context: Agent
    angles: Agent
    drafts: Agent
    selection: Agent


def build_openai_client() -> OpenAI:
    return OpenAI(api_key=settings.api_key, base_url=settings.base_url)


def build_openai_model(model_name: str) -> OpenAIChatModel:
    provider = OpenAIProvider(api_key=settings.api_key, base_url=settings.base_url)
    return OpenAIChatModel(model_name, provider=provider)


def get_agents(model_name: str) -> Agents:
    model = build_openai_model(model_name)

    template_agent = Agent(
        model=model,
        output_type=str,
        system_prompt=TEMPLATE_SYSTEM_PROMPT,
    )
    context_agent = Agent(
        model=model,
        output_type=str,
        system_prompt=CONTEXT_SYSTEM_PROMPT,
    )
    angles_agent = Agent(
        model=model,
        output_type=str,
        system_prompt=ANGLES_SYSTEM_PROMPT,
    )
    draft_agent = Agent(
        model=model,
        output_type=str,
        system_prompt=DRAFTS_SYSTEM_PROMPT,
    )
    selection_agent = Agent(
        model=model,
        output_type=str,
        system_prompt=SELECTION_SYSTEM_PROMPT,
    )

    return Agents(
        template=template_agent,
        context=context_agent,
        angles=angles_agent,
        drafts=draft_agent,
        selection=selection_agent,
    )
