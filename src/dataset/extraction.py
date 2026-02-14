from pathlib import Path

import yaml
from openai import OpenAI
from pydantic import BaseModel, Field

from src.paths import CONFIG_DIR
from src.settings import settings
from src.templates import environment


class ExtractionConfig(BaseModel):
    model: str
    temperature: float = 0.2
    max_completion_tokens: int = 300


class SeedExtractionResult(BaseModel):
    seeds: list[str] = Field(min_length=1, max_length=3)


class ExtractionPipeline:
    def __init__(self, config_path: Path | None = None) -> None:
        if config_path is None:
            config_path = CONFIG_DIR / "extraction.yaml"

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config = ExtractionConfig.model_validate(payload)

        self.config = config
        self.template = environment.get_template("extraction.j2")

        self.client = OpenAI(
            base_url=settings.OPENAI_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
            timeout=60,
        )

    def run(self, joke: str) -> SeedExtractionResult:
        user_prompt = self.template.render(joke=joke.strip())
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "seed_extraction_result",
                    "schema": SeedExtractionResult.model_json_schema(),
                    "strict": True,
                },
            },
            temperature=self.config.temperature,
            max_completion_tokens=self.config.max_completion_tokens,
        )

        print(response.to_json())

        content = response.choices[0].message.content

        if not content:
            raise ValueError("Model returned empty content.")

        return SeedExtractionResult.model_validate_json(content)
