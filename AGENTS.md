# AGENTS.md

This file provides guidance for AI coding agents (GitHub Copilot, Cursor, etc.) working on the **audio-cleaner** project.

## Project Overview

`audio-cleaner` is a Python library and CLI tool that processes FLAC and WAV audio files to:

1. **Improve poor audio quality** — normalisation, dynamic-range compression, equalisation, de-clipping.
2. **Detect and remove ads / interrupts** — silence detection, audio fingerprinting, segment classification.

## Repository Layout

```
audio-cleaner/
├── src/
│   └── audio_cleaner/       # Main package
│       ├── __init__.py
│       ├── __main__.py      # CLI entry-point
│       ├── quality.py       # Audio quality improvement
│       └── ads.py           # Ad / interrupt detection & removal
├── tests/                   # pytest test suite
│   └── test_*.py
├── docs/                    # Planning & design documents
│   └── plan-audio-quality.md
├── pyproject.toml           # Project metadata, ruff, pyrefly, pytest config
├── AGENTS.md                # ← you are here
└── README.md
```

## Toolchain

| Tool       | Purpose                        | Command                     |
|------------|--------------------------------|-----------------------------|
| `uv`       | Package & virtual-env manager  | `uv sync` / `uv run`        |
| `ruff`     | Linter + code formatter        | `ruff check src tests`      |
| `pyrefly`  | Static type checker            | `pyrefly check src`         |
| `pytest`   | Test runner                    | `uv run pytest`             |

### Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install all dependencies
uv sync --extra dev

# Verify the setup
uv run audio-cleaner --help
```

### Running quality checks

```bash
# Format & lint
ruff format src tests
ruff check src tests

# Type check
pyrefly check src

# Tests
uv run pytest
```

## Coding Standards

- **Python ≥ 3.12** — use modern syntax (f-strings, `match`, structural pattern matching, `|` union types, etc.).
- **Type annotations** — every public function and method must have full type annotations.
- **Docstrings** — use Google-style docstrings for all public symbols.
- **Line length** — maximum 100 characters (enforced by ruff).
- **Imports** — stdlib → third-party → first-party, separated by blank lines (enforced by ruff/isort).
- **No bare `except`** — always catch specific exception types.
- **Pure Python preferred** — prefer `scipy`, `numpy`, and `soundfile` over heavy external binaries.

## Supported Audio Formats

| Format | Extension | Notes                            |
|--------|-----------|----------------------------------|
| FLAC   | `.flac`   | Lossless, primary target format  |
| WAV    | `.wav`    | Uncompressed PCM                 |

All audio is loaded as float32 numpy arrays via `soundfile` or `librosa`.  
Sample rate is preserved unless the user explicitly requests resampling.

## Key Design Decisions

1. **Pure-Python first** — avoid shell-outs to `ffmpeg` or other binaries unless no Python alternative exists.
2. **Non-destructive by default** — original files are never overwritten; outputs go to a user-specified directory.
3. **Pipeline architecture** — cleaning steps are composable so users can chain them (e.g., normalise → remove ads).
4. **Batch processing** — the CLI accepts a directory or glob pattern as input.
5. **Reproducible** — all random seeds and configurable parameters are exposed via `pyproject.toml` or CLI flags.

## PR Guidelines

- Keep PRs focused: one feature / fix per PR.
- All new code must have corresponding tests in `tests/`.
- Run `ruff check`, `pyrefly check`, and `pytest` before opening a PR.
- PR titles should follow Conventional Commits: `feat:`, `fix:`, `docs:`, `chore:`.
- Reference the relevant planning document (e.g., `docs/plan-audio-quality.md`) in the PR description.
