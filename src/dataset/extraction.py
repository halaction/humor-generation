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
    max_completion_tokens: int


class NormalizeResult(BaseModel):
    keywords: list[str] = Field(min_length=1, max_length=3)


class ExtractResult(BaseModel):
    substrings: list[str] = Field(min_length=1, max_length=3)


class ExtractionPipeline:
    def __init__(self, config_path: Path | None = None) -> None:
        if config_path is None:
            config_path = CONFIG_DIR / "extraction.yaml"

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = ExtractionConfig.model_validate(payload)

        self.config = config
        self.extract_system_template = environment.get_template("extract_system.j2")
        self.extract_user_template = environment.get_template("extract_user.j2")

        self.normalize_system_template = environment.get_template("normalize_system.j2")
        self.normalize_user_template = environment.get_template("normalize_user.j2")

        self.client = OpenAI(
            base_url=settings.OPENAI_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
            timeout=60,
        )

    def _complete_with_schema(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_model: type[BaseModel],
        schema_name: str,
    ) -> BaseModel:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_completion_tokens=self.config.max_completion_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema_model.model_json_schema(),
                },
            },
        )

        if not response.choices:
            raise ValueError("Model returned empty choices.")

        message = response.choices[0].message
        if not message.content:
            raise ValueError("Model returned empty content.")

        content = json.loads(message.content)
        return schema_model.model_validate(content)

    def run(self, joke: str) -> BaseModel:
        joke = joke.strip()

        extract_result = self._complete_with_schema(
            system_prompt=self.extract_system_template.render(),
            user_prompt=self.extract_user_template.render(joke=joke),
            schema_model=ExtractResult,
            schema_name="extract_result",
        )

        print(extract_result)

        substrings = [substring.strip() for substring in extract_result.substrings]

        return self._complete_with_schema(
            system_prompt=self.normalize_system_template.render(),
            user_prompt=self.normalize_user_template.render(substrings=substrings),
            schema_model=NormalizeResult,
            schema_name="normalize_result",
        )
