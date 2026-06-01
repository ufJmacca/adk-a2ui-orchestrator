# ADK A2UI Orchestrator Demo

Local Python-first ADK 2.0+ A2UI orchestrator demo for business banking
relationship managers.

## Project Baseline

This repository is managed with `uv`. Use `uv` for dependency management,
lockfile updates, virtual environment synchronization, and command execution.

```bash
uv sync --locked
uv run pytest
uv run ruff check .
```

Runtime secrets must be provided through environment variables or a local
`.env` file copied from `.env.example`. Do not commit real secrets.
