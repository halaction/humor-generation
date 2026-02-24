import asyncio
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
import yaml
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.paths import EXTRACTION_CONFIG_PATH
from src.paths import EXTRACTION_RESULTS_PATH
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


class ExtractionRecord(BaseModel):
    joke_id: str
    keywords: list[str]


ExtractionInputs = Dataset
ExtractionOutputs = list[ExtractionRecord]

class ExtractionPipeline:
    def __init__(self, config_path: Path | None = None) -> None:
        if config_path is None:
            config_path = EXTRACTION_CONFIG_PATH

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

    async def _process_joke(self, joke_id: str, joke: str, semaphore: asyncio.Semaphore) -> ExtractionRecord:
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

        return ExtractionRecord(joke_id=joke_id, keywords=normalize_result.keywords)

    @staticmethod
    def _load_existing_results(path: Path) -> dict[str, list[str]]:
        if not path.exists():
            return {}

        records: dict[str, list[str]] = {}
        with path.open(encoding="utf-8") as input_file:
            for line in input_file:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                joke_id = str(payload["joke_id"])
                keywords = [str(item).strip() for item in payload["keywords"]]
                records[joke_id] = keywords

        return records

    @staticmethod
    def _sort_joke_id(value: str) -> tuple[int, Any]:
        return (0, int(value)) if value.isdigit() else (1, value)

    def _save_results(self, results: ExtractionOutputs, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        existing = self._load_existing_results(output_path)

        for item in results:
            existing[item.joke_id] = item.keywords

        with output_path.open("w", encoding="utf-8") as output_file:
            for joke_id in sorted(existing, key=self._sort_joke_id):
                payload = {"joke_id": joke_id, "keywords": existing[joke_id]}
                output_file.write(f"{json.dumps(payload, ensure_ascii=False)}\n")

        return output_path

    async def run(
        self,
        inputs: ExtractionInputs,
        output_path: Path = EXTRACTION_RESULTS_PATH,
    ) -> tuple[ExtractionOutputs, Path]:
        if "text" not in inputs.column_names:
            raise ValueError("Dataset must contain a 'text' column.")

        if "id" not in inputs.column_names:
            raise ValueError("Dataset must contain an 'id' column.")

        semaphore = asyncio.Semaphore(self.config.max_parallel_requests)
        tasks = [
            self._process_joke(joke_id=str(row["id"]), joke=row["text"], semaphore=semaphore) for row in inputs
        ]
        results = await asyncio.gather(*tasks)
        saved_path = self._save_results(results=results, output_path=output_path)
        return results, saved_path
