import logging

from pydantic import BaseModel
from pydantic_ai import Agent

from agent.agents import Agents
from agent.config import Config
from agent.logging import RunLogger
from agent.models import RunSummary
from agent.prompts import PromptRenderer


def run_pipeline(agents: Agents, config: Config, run_logger: RunLogger) -> RunSummary:
    logger = logging.getLogger(__name__)
    logger.info("starting pipeline with model=%s k=%s", config.model, config.k)

    prompt_renderer = PromptRenderer()
    input_text = config.input or ""

    template_prompt = prompt_renderer.template_prompt(input_text)
    template = _run_agent(agents.template, template_prompt)
    run_logger.log_step("template_analysis", template, {"prompt": template_prompt})

    context_prompt = prompt_renderer.context_prompt(input_text, template)
    context = _run_agent(agents.context, context_prompt)
    run_logger.log_step("context_analysis", context, {"prompt": context_prompt})

    association_prompt = prompt_renderer.association_prompt(config.k, template, context)
    associations = _run_agent(agents.associations, association_prompt)
    run_logger.log_step("associations", associations, {"prompt": association_prompt})

    draft_prompt = prompt_renderer.draft_prompt(config.k, template, context, associations)
    drafts = _run_agent(agents.drafts, draft_prompt)
    run_logger.log_step("drafts", drafts, {"prompt": draft_prompt})

    judge_prompt = prompt_renderer.judge_prompt(template, context, drafts)
    selection = _run_agent(agents.judge, judge_prompt)
    run_logger.log_step("selection", selection, {"prompt": judge_prompt})

    return RunSummary(
        template_analysis=template,
        context_analysis=context,
        associations=associations,
        drafts=drafts,
        selection=selection,
    )


def _run_agent(agent: Agent, prompt: str) -> BaseModel:
    result = agent.run_sync(prompt)
    if isinstance(result, BaseModel):
        return result
    for attr in ("data", "output", "result"):
        if hasattr(result, attr):
            value = getattr(result, attr)
            if isinstance(value, BaseModel):
                return value
            return value
    raise TypeError(f"Unexpected agent result type: {type(result).__name__}")
