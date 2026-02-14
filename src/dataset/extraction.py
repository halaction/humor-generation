from pathlib import Path

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


class SeedExtractionResult(BaseModel):
    seeds: list[str] = Field(min_length=1, max_length=3)


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

    def run(self, joke: str) -> SeedExtractionResult:
        system_prompt = self.system_template.render()
        user_prompt = self.user_template.render(joke=joke.strip())

        response = self.client.responses.parse(
            model=self.config.model,
            input=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            text_format=SeedExtractionResult,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
        )

        print(response.to_json())

        content = response.output_parsed

        if not content:
            raise ValueError("Model returned empty content.")

        # return SeedExtractionResult.model_validate_json(content)
        return content
