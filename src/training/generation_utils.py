from __future__ import annotations


def extract_completion_ids(row_ids: list[int], *, input_width: int, pad_token_id: int) -> list[int]:
    completion = row_ids[input_width:]
    return [token for token in completion if token != pad_token_id]
