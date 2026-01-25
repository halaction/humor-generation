import json
import logging
import re
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent

from agent.agents import Agents
from agent.config import Config
from agent.logging import RunLogger
from agent.models import RunSummary
from agent.prompts import PromptRenderer

_NUMBERED_ITEM_RE = re.compile(r"^(\d+)\s*[\).:-]\s+(.*)$")
_BULLET_ITEM_RE = re.compile(r"^[-*+]\s+(.*)$")


def run_pipeline(agents: Agents, config: Config, run_logger: RunLogger) -> RunSummary:
    logger = logging.getLogger(__name__)
    logger.info("starting pipeline with model=%s k=%s", config.model, config.k)

    prompt_renderer = PromptRenderer()
    constraint = config.input or ""

    template_prompt = prompt_renderer.template_prompt(constraint)
    template_raw, template_reasoning = _run_agent(agents.template, template_prompt)
    template = _coerce_text(template_raw)
    run_logger.log_step(
        "template",
        {"content": template, "reasoning_content": template_reasoning},
        {"prompt": template_prompt},
    )

    context_prompt = prompt_renderer.context_prompt(constraint)
    context_raw, context_reasoning = _run_agent(agents.context, context_prompt)
    context = _coerce_text(context_raw)
    run_logger.log_step(
        "context",
        {"content": context, "reasoning_content": context_reasoning},
        {"prompt": context_prompt},
    )

    angles_prompt = prompt_renderer.angles_prompt(config.k, context)
    angles_raw, angles_reasoning = _run_agent(agents.angles, angles_prompt)
    angles = _coerce_list("angles", angles_raw, config.k)
    run_logger.log_step(
        "angles",
        {"content": angles, "reasoning_content": angles_reasoning},
        {"prompt": angles_prompt},
    )

    drafts_prompt = prompt_renderer.drafts_prompt(config.k, template, _format_numbered_list(angles))
    drafts_raw, drafts_reasoning = _run_agent(agents.drafts, drafts_prompt)
    drafts = _coerce_list("drafts", drafts_raw, config.k)
    run_logger.log_step(
        "drafts",
        {"content": drafts, "reasoning_content": drafts_reasoning},
        {"prompt": drafts_prompt},
    )

    selection_prompt = prompt_renderer.selection_prompt(constraint, _format_numbered_list(drafts))
    selection_raw, selection_reasoning = _run_agent(agents.selection, selection_prompt)
    selection = _coerce_text(selection_raw)
    run_logger.log_step(
        "selection",
        {"content": selection, "reasoning_content": selection_reasoning},
        {"prompt": selection_prompt},
    )

    return RunSummary(
        template=template,
        context=context,
        angles=angles,
        drafts=drafts,
        selection=selection,
    )


def _run_agent(agent: Agent, prompt: str) -> tuple[Any, str | None]:
    result = agent.run_sync(prompt)
    reasoning = _extract_reasoning_content(result)
    output = _extract_output(result)
    return output, reasoning


def _extract_output(result: Any) -> Any:
    if isinstance(result, BaseModel):
        return result
    for attr in ("output", "data", "result"):
        if hasattr(result, attr):
            value = getattr(result, attr)
            if isinstance(value, BaseModel):
                return value
            return value
    return result


def _extract_reasoning_content(result: Any) -> str | None:
    try:
        response = getattr(result, "response")
    except Exception:
        response = None
    if response is None:
        return None
    thinking = getattr(response, "thinking", None)
    if isinstance(thinking, str) and thinking.strip():
        return thinking
    provider_details = getattr(response, "provider_details", None)
    if isinstance(provider_details, dict):
        for key in ("reasoning_content", "reasoning"):
            value = provider_details.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    return str(value).strip()


def _coerce_list(step: str, raw: Any, expected: int) -> list[str]:
    items = _parse_list_output(raw)
    cleaned = [_strip_leading_label(item) for item in items]
    cleaned = [item for item in cleaned if item]
    if expected and len(cleaned) != expected:
        logging.getLogger(__name__).warning("%s expected %s items, got %s", step, expected, len(cleaned))
    if expected and len(cleaned) > expected:
        cleaned = cleaned[:expected]
    return cleaned


def _parse_list_output(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if raw is None:
        return []

    text = str(raw).strip()
    if not text:
        return []

    text = _strip_code_fence(text)

    if text.startswith("["):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]

    items: list[str] = []
    buffer: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        numbered_match = _NUMBERED_ITEM_RE.match(stripped)
        if numbered_match:
            if buffer:
                items.append(" ".join(buffer).strip())
                buffer = []
            buffer.append(numbered_match.group(2).strip())
            continue
        bullet_match = _BULLET_ITEM_RE.match(stripped)
        if bullet_match:
            if buffer:
                items.append(" ".join(buffer).strip())
                buffer = []
            buffer.append(bullet_match.group(1).strip())
            continue
        if buffer:
            buffer.append(stripped)
        else:
            buffer.append(stripped)
    if buffer:
        items.append(" ".join(buffer).strip())
    return items


def _strip_leading_label(text: str) -> str:
    stripped = text.strip()
    numbered_match = _NUMBERED_ITEM_RE.match(stripped)
    if numbered_match:
        return numbered_match.group(2).strip()
    bullet_match = _BULLET_ITEM_RE.match(stripped)
    if bullet_match:
        return bullet_match.group(1).strip()
    return stripped


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 2:
        return text
    if not lines[-1].strip().startswith("```"):
        return text
    return "\n".join(lines[1:-1]).strip()


def _format_numbered_list(items: list[str]) -> str:
    if not items:
        return ""
    lines = []
    for index, item in enumerate(items, start=1):
        normalized = " ".join(item.split())
        lines.append(f"{index}. {normalized}")
    return "\n".join(lines)
