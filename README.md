# humor-generation

Research-grade, transparent humor generation MVP with a multi-stage LLM pipeline.

## Requirements

- Python 3.13+
- `OPENAI_API_KEY` set in the environment

Optional environment variables:

- `OPENAI_BASE_URL` to target OpenRouter or other compatible endpoints
- `OPENAI_MODEL` for a default model name (overridden by `--model`)

## Usage

```bash
python3 -m agent --model "openai/gpt-4o-mini" --input "banana fish" --k 4
```

Run artifacts and logs are written to `runs/latest/`.
