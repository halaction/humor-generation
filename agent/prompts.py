import json

from jinja2 import BaseLoader, Environment
from pydantic import BaseModel

TEMPLATE_SYSTEM_PROMPT = """You are a humor-generation analyst.
Your task is to interpret the input as a joke constraint and infer a reusable template.
Return short, concrete bullet reasoning. Avoid chain-of-thought narration.
Keep jokes safe and avoid hateful, harassing, or targeted content.
"""


CONTEXT_SYSTEM_PROMPT = """You are a context analyst for humor generation.
Ground reasoning in the provided input and template analysis.
Return short, concrete bullet reasoning. Avoid chain-of-thought narration.
"""


ASSOCIATION_SYSTEM_PROMPT = """You generate diverse narrative associations for humor.
Each association must be distinct and tightly linked to the context analysis.
Keep outputs compact and focused.
"""


DRAFT_SYSTEM_PROMPT = """You draft concise jokes from narrative associations.
Each joke should be a one-liner and traceable to one association.
Keep jokes safe and avoid hateful, harassing, or targeted content.
"""


JUDGE_SYSTEM_PROMPT = """You are a rubric-based humor judge.
Score each joke on readability, novelty, relevance, and funniness (1-5).
Provide brief notes per joke, then select the best overall.
"""


_TEMPLATE_PROMPT = """Input:
{{ input_text }}

Return TemplateAnalysis with:
- input_summary
- input_type (none/topic/headline/constraint/other)
- constraint (normalized, explicit)
- template (short description of the joke format)
- humor_mechanics (3-6 items)
- tone (2-4 items)
- avoidances (safety notes)
- reasoning_notes (short bullets)
If no input is provided, choose a safe everyday topic and note it in constraint.
"""


_CONTEXT_PROMPT = """Input:
{{ input_text }}

Template analysis JSON:
{{ template_json }}

Return ContextAnalysis with:
- topic_summary
- key_facts
- cultural_context
- comedic_angles
- possible_targets
- risk_notes
- reasoning_notes
"""


_ASSOCIATION_PROMPT = """Template analysis JSON:
{{ template_json }}

Context analysis JSON:
{{ context_json }}

Generate exactly {{ k }} associations. Use association_id A1..A{{ k }}.
Return AssociationSet with associations containing:
- association_id
- title
- premise
- connection
- twist
"""


_DRAFT_PROMPT = """Template analysis JSON:
{{ template_json }}

Context analysis JSON:
{{ context_json }}

Associations JSON:
{{ associations_json }}

Draft exactly {{ k }} jokes. Use joke_id J1..J{{ k }}.
Each joke must reference one association_id from the list.
Return JokeDraftSet with jokes containing:
- joke_id
- text
- angle
- rationale
"""


_JUDGE_PROMPT = """Rubric:
- readability: clear and easy to parse
- novelty: unexpected or fresh angle
- relevance: fits the input constraints
- funniness: comedic impact

Template analysis JSON:
{{ template_json }}

Context analysis JSON:
{{ context_json }}

Drafts JSON:
{{ drafts_json }}

Return JokeSelection with:
- best_id (must match a joke_id)
- best_text (verbatim from drafts)
- rubric_summary
- scores (one per joke_id)
- improvement_suggestions
"""


class PromptRenderer:
    """Render user prompts using Jinja2 templates."""

    def __init__(self) -> None:
        self._env = Environment(
            loader=BaseLoader(),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._templates = {
            "template": self._env.from_string(_TEMPLATE_PROMPT),
            "context": self._env.from_string(_CONTEXT_PROMPT),
            "associations": self._env.from_string(_ASSOCIATION_PROMPT),
            "drafts": self._env.from_string(_DRAFT_PROMPT),
            "judge": self._env.from_string(_JUDGE_PROMPT),
        }

    def template_prompt(self, input_text: str) -> str:
        """Render the template-level analysis prompt."""

        return self._render("template", input_text=_normalize_input(input_text))

    def context_prompt(self, input_text: str, template: BaseModel) -> str:
        """Render the context analysis prompt."""

        return self._render(
            "context",
            input_text=_normalize_input(input_text),
            template_json=_to_json(template),
        )

    def association_prompt(self, k: int, template: BaseModel, context: BaseModel) -> str:
        """Render the association generation prompt."""

        return self._render(
            "associations",
            k=k,
            template_json=_to_json(template),
            context_json=_to_json(context),
        )

    def draft_prompt(self, k: int, template: BaseModel, context: BaseModel, associations: BaseModel) -> str:
        """Render the joke drafting prompt."""

        return self._render(
            "drafts",
            k=k,
            template_json=_to_json(template),
            context_json=_to_json(context),
            associations_json=_to_json(associations),
        )

    def judge_prompt(self, template: BaseModel, context: BaseModel, drafts: BaseModel) -> str:
        """Render the rubric-based judging prompt."""

        return self._render(
            "judge",
            template_json=_to_json(template),
            context_json=_to_json(context),
            drafts_json=_to_json(drafts),
        )

    def _render(self, name: str, **context: object) -> str:
        template = self._templates[name]
        return template.render(**context)


def _normalize_input(input_text: str) -> str:
    if input_text.strip():
        return input_text
    return "[NO INPUT PROVIDED]"


def _to_json(model: BaseModel) -> str:
    """Serialize a Pydantic model to JSON for prompts."""

    return json.dumps(model.model_dump(), ensure_ascii=True, indent=2, sort_keys=True)
