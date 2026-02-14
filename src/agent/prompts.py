from jinja2 import BaseLoader, Environment

TEMPLATE_SYSTEM_PROMPT = """You are a comedy writer. 
Your job is to turn a user constraint into a single, clear, imperative writing instruction a comedian could follow.
Do NOT write jokes yet. Be specific, operational, and short.
"""

CONTEXT_SYSTEM_PROMPT = """You are a comedy writer. 
You explore meanings, cultural links, wordplay opportunities, stereotypes, and surprising connections.
Be expansive and exploratory. It's okay to think step-by-step. Do NOT draft jokes yet.
"""

ANGLES_SYSTEM_PROMPT = """You are a comedy writer.
Your task is brainstorming angles. 
Generate diverse, non-overlapping angles that could plausibly yield a strong punchline.
You MUST output a list of texts only.
"""

DRAFTS_SYSTEM_PROMPT = """You are a comedy writer.
You write finished jokes with punchlines, not premises. 
Follow the given template strictly.
You MUST output a list of texts only.
"""

SELECTION_SYSTEM_PROMPT = """You are a strict annotator.
Your task is to evaluate jokes.
You will be given:
- the original constraint
- a list of drafted jokes
Rate the jokes according to given criteria, think step by step.
Output ONLY the winning joke text.
"""


TEMPLATE_USER_PROMPT = """<constraint>
{{ constraint }}
</constraint>

Write ONE imperative template instruction (1-4 sentences). It must specify:
- the joke format (one-liner / two-liner / news headline parody / etc.)
- hard constraints (exact words to include, taboo level, POV, length limit)

Output only the template instruction, nothing else.
"""

CONTEXT_USER_PROMPT = """<constraint>
{{ constraint }}
</constraint>

Write a prewriting exploration with headings (plain text):
- Literal decoding (what is being asked, what must be included)
- Word / phrase analysis (meanings, double meanings, nearby phrases, collocations, homophones)
- Cultural hooks (memes, current / historical context, stereotypes, common tropes)
- Tension points (what could be incongruous, benign violation, status reversals, anxieties)

No jokes. No lists of final outputs yet -- just exploration.
"""

ANGLES_USER_PROMPT = """<context>
{{ context }}
</context>

Generate exactly {{ k }} ANGLES as a numbered list.
Rules:
- Each item is 1-2 lines max.
- Each item must be an angle / narrative / relationship / association, not a joke.
- Each item can use a comedic mechanism like misdirection, irony, analogy, benign violation, wordplay, etc.
- Avoid duplicates; force different contexts.
"""

DRAFTS_USER_PROMPT = """<template>
{{ template }}
</template>

<angles>
{{ angles }}
</angles>

Write {{ k }} drafts, one per angle, as a numbered list 1..{{ k }}.
Rules:
- Must have a clear punchline, expanding the respective angle.
- Must adhere to the given template.
- Each draft must be 1-2 sentences max, unless the template demands otherwise.
- No explanations, no labels, no commentary -- just the joke text per line.
"""

SELECTION_USER_PROMPT = """<constraint>
{{ constraint }}
</constraint>

<drafts>
{{ drafts }}
</drafts>

Rate the jokes according to this rubric (in this order):
1) Adheres perfectly to the constraint (especially exact words/themes/format).
2) Feels human-written (not "template-y", not generic, not vibe-y, not obviously LLM).
3) Novelty / originality (not a common stock joke).
4) Funny to a general audience (at least one clear humor mechanism: incongruity, irony, reversal, benign violation, wordplay, etc.)

Output ONLY the winning joke text. Nothing else.
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
            "template": self._env.from_string(TEMPLATE_USER_PROMPT),
            "context": self._env.from_string(CONTEXT_USER_PROMPT),
            "angles": self._env.from_string(ANGLES_USER_PROMPT),
            "drafts": self._env.from_string(DRAFTS_USER_PROMPT),
            "selection": self._env.from_string(SELECTION_USER_PROMPT),
        }

    def template_prompt(self, constraint: str) -> str:
        """Render the template prompt."""

        return self._render("template", constraint=_normalize_input(constraint))

    def context_prompt(self, constraint: str) -> str:
        """Render the context prompt."""

        return self._render("context", constraint=_normalize_input(constraint))

    def angles_prompt(self, k: int, context: str) -> str:
        """Render the angle brainstorming prompt."""

        return self._render("angles", k=k, context=_normalize_input(context))

    def drafts_prompt(self, k: int, template: str, angles: str) -> str:
        """Render the joke drafting prompt."""

        return self._render("drafts", k=k, template=_normalize_input(template), angles=_normalize_input(angles))

    def selection_prompt(self, constraint: str, drafts: str) -> str:
        """Render the selection prompt."""

        return self._render("selection", constraint=_normalize_input(constraint), drafts=_normalize_input(drafts))

    def _render(self, name: str, **context: object) -> str:
        template = self._templates[name]
        return template.render(**context)


def _normalize_input(input_text: str) -> str:
    if input_text.strip():
        return input_text
    return "[NO INPUT PROVIDED]"
