import asyncio
import json
from pathlib import Path

from datasets import Dataset
import yaml
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.paths import CONFIG_DIR
from src.settings import settings
from src.templates import environment


class ExtractionConfig(BaseModel):
    model: str
    temperature: float
    max_completion_tokens: int
    max_parallel_requests: int = Field(gt=0)


class ExtractResult(BaseModel):
    substrings: list[str] = Field(min_length=1, max_length=3)


class NormalizeResult(BaseModel):
    keywords: list[str] = Field(min_length=1, max_length=3)


ExtractionInputs = Dataset
ExtractionOutputs = list[NormalizeResult]


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

        self.client = AsyncOpenAI(
            base_url=settings.OPENAI_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
            timeout=60,
        )

    async def _complete_with_schema(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_model: type[BaseModel],
        schema_name: str,
    ) -> BaseModel:
        response = await self.client.chat.completions.create(
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

    async def _process_joke(self, joke: str, semaphore: asyncio.Semaphore) -> NormalizeResult:
        async with semaphore:
            joke = str(joke).strip()
            extract_result = await self._complete_with_schema(
                system_prompt=self.extract_system_template.render(),
                user_prompt=self.extract_user_template.render(joke=joke),
                schema_model=ExtractResult,
                schema_name="extract_result",
            )

            substrings = [substring.strip() for substring in extract_result.substrings]
            normalize_result = await self._complete_with_schema(
                system_prompt=self.normalize_system_template.render(),
                user_prompt=self.normalize_user_template.render(substrings=substrings),
                schema_model=NormalizeResult,
                schema_name="normalize_result",
            )

        return normalize_result

    async def run(self, inputs: ExtractionInputs) -> ExtractionOutputs:
        if "text" not in inputs.column_names:
            raise ValueError("Dataset must contain a 'text' column.")

        semaphore = asyncio.Semaphore(self.config.max_parallel_requests)
        tasks = [self._process_joke(joke=row["text"], semaphore=semaphore) for row in inputs]
        return await asyncio.gather(*tasks)
