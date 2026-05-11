from src.training.generation_utils import extract_completion_ids


def test_extract_completion_ids_uses_input_width_not_nonpad_len() -> None:
    # Left-padded prompt + generated tokens.
    # Prompt occupies width 6 with one leading pad.
    row = [0, 11, 12, 13, 14, 15, 91, 92, 0]
    completion = extract_completion_ids(row, input_width=6, pad_token_id=0)
    assert completion == [91, 92]


def test_extract_completion_ids_drops_padding_tokens() -> None:
    row = [0, 21, 22, 23, 77, 0, 0]
    completion = extract_completion_ids(row, input_width=4, pad_token_id=0)
    assert completion == [77]
