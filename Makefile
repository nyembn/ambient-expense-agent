.PHONY: install run playground test lint clean generate-traces grade

install:
	uv sync

run:
	uv run uvicorn app.fast_api_app:app --host 0.0.0.0 --port 8080

playground:
	uv run uvicorn app.fast_api_app:app --host 0.0.0.0 --port 8080

test:
	uv run pytest tests/unit tests/integration

lint:
	uv run ruff check .

clean:
	rm -rf .venv __pycache__ */__pycache__ */*/__pycache__ .pytest_cache

generate-traces:
	uv run python tests/eval/generate_traces.py

grade:
	export PATH="$(HOME)/.local/bin:$$PATH"; agents-cli eval grade --traces artifacts/traces/generated_traces.json --config tests/eval/eval_config.yaml
