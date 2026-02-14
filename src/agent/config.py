from dataclasses import dataclass
from typing import Any

from agent.settings import settings


@dataclass(frozen=True)
class Config:
    model: str
    input: str | None
    k: int


def get_config(args: Any) -> Config:
    model = args.model or settings.model

    return Config(
        model=model,
        input=args.input,
        k=args.k,
    )
