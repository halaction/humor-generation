import json
from pathlib import Path
from typing import Literal

import yaml
from openai import OpenAI
from pydantic import BaseModel, Field

from src.paths import CONFIG_DIR
from src.settings import settings
from src.templates import environment


class ExtractionConfig(BaseModel):
    model: str
    temperature: float
    max_output_tokens: int
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"]
    reasoning_summary: Literal["auto", "concise", "detailed"]


class ExtractionResult(BaseModel):
    keywords: list[str] = Field(min_length=1, max_length=3)


class ExtractionPipeline:
    def __init__(self, config_path: Path | None = None) -> None:
        if config_path is None:
            config_path = CONFIG_DIR / "extraction.yaml"

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = ExtractionConfig.model_validate(payload)

        self.config = config
        self.system_template = environment.get_template("extraction_system.j2")
        self.user_template = environment.get_template("extraction_user.j2")

        self.client = OpenAI(
            base_url=settings.OPENAI_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
            timeout=60,
        )

    def run(self, joke: str) -> ExtractionResult:
        system_prompt = self.system_template.render()
        user_prompt = self.user_template.render(joke=joke.strip())

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_completion_tokens=self.config.max_output_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction_result",
                    "strict": True,
                    "schema": ExtractionResult.model_json_schema(),
                },
            },
            # extra_body={
            #     "reasoning": {
            #         "effort": self.config.reasoning_effort,
            #         "summary": self.config.reasoning_summary,
            #     },
            # },
        )

        print()
        print(response.usage)

        if not response.choices:
            raise ValueError("Model returned empty content.")

        message = response.choices[0].message
        if not message.content:
            raise ValueError("Model returned empty content.")

        print(message)

        content = json.loads(message.content)
        return ExtractionResult.model_validate(content)
