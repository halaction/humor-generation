from __future__ import annotations

import re


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    cleaned = _THINK_RE.sub("", text)
    return cleaned.strip()
