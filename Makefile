.PHONY: test lint data smoke run full

test:
	uv run pytest -q

lint:
	uvx ruff check .

# One parquet shard (~500 MB, ~138 queries), de-obfuscated into data/. Never commit data/.
data:
	uv run python -m bcp.data --queries 50

smoke:
	uv run python -m bcp.run --n 5 --out out/smoke

run:
	uv run python -m bcp.run --n 50 --out out/

# All 830 queries, official Qwen3-32B judge; resumable (re-run the same command after a crash).
full:
	uv run python -m bcp.agent --model deepseek/deepseek-v4-flash-0731 --all --workers 8 --out out/full
