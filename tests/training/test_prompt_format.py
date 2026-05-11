from src.training.prompt_format import strip_thinking


def test_strip_thinking_removes_think_block() -> None:
    text = "<think>internal</think>\nFinal joke."
    assert strip_thinking(text) == "Final joke."
