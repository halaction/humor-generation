import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def configure_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


class RunLogger:
    def __init__(self, run_dir: Path) -> None:
        self._path = run_dir / "artifacts.jsonl"

    def log_step(self, step: str, payload: BaseModel | dict[str, Any], meta: dict[str, Any] | None = None) -> None:
        record: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "step": step,
            "payload": self._serialize(payload),
        }
        if meta:
            record["meta"] = meta
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        logging.getLogger(__name__).info("logged step=%s", step)

    @staticmethod
    def _serialize(payload: BaseModel | dict[str, Any]) -> dict[str, Any]:
        if isinstance(payload, BaseModel):
            return payload.model_dump()
        return payload
