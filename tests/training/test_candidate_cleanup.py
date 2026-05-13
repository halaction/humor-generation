import pytest

pytest.importorskip("torch")
pytest.importorskip("pyarrow")
pytest.importorskip("datasets")

from scripts.generate_checkpoint_candidates import _candidate_quality_summary, _clean_candidate_text
from src.models import CandidateOutput


def test_closed_thinking_cleanup_preserves_final_answer() -> None:
    text = "<think>plan words</think>\n\nWhy did the banana call a lawyer? It got peeled."
    cleaned = _clean_candidate_text(text, strip_thinking=True)
    assert "<think>" not in cleaned
    assert cleaned == "Why did the banana call a lawyer? It got peeled."


def test_unclosed_thinking_cleanup_does_not_leak_trace() -> None:
    text = "<think>Okay, I need a joke about ducks and taxes. Maybe"
    cleaned = _clean_candidate_text(text, strip_thinking=True)
    assert cleaned == ""


def test_candidate_cleanup_removes_wrappers_and_suffixes() -> None:
    text = "Sure! Here's a joke using the keywords cat, bill:\n\nWhy did the cat pay rent? It had a purr-lease.\n\nNote: This is a pun."
    cleaned = _clean_candidate_text(text, strip_thinking=True)
    assert cleaned == "Why did the cat pay rent? It had a purr-lease."


def test_candidate_quality_summary_counts_bad_outputs() -> None:
    rows = [
        CandidateOutput(id=1, keywords=["cat"], model="m", text=""),
        CandidateOutput(id=2, keywords=["dog"], model="m", text="<think>bad"),
        CandidateOutput(id=3, keywords=["bird"], model="m", text="Sure! Here's a joke. Note: x"),
    ]
    summary = _candidate_quality_summary(rows)
    assert summary["rows"] == 3
    assert summary["empty_text_count"] == 1
    assert summary["contains_think_count"] == 1
    assert summary["contains_wrapper_count"] == 1
    assert summary["contains_note_count"] == 1
