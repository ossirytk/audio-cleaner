# audio-cleaner

A Python toolkit for cleaning **FLAC** and **WAV** audio files.

## Features

| Feature | Module | Plan |
|---|---|---|
| Loudness normalisation, de-clipping, EQ, compression | `audio_cleaner.quality` | [docs/plan-audio-quality.md](docs/plan-audio-quality.md) |
| Background noise removal (spectral gating, Wiener, notch filters) | `audio_cleaner.noise` | [docs/plan-noise-removal.md](docs/plan-noise-removal.md) |
| Ad / interrupt detection and removal | `audio_cleaner.ads` | [docs/plan-ad-detection.md](docs/plan-ad-detection.md) |

## Quick Start

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run the CLI
uv run audio-cleaner --help
```

## Toolchain

- **[uv](https://github.com/astral-sh/uv)** — fast package & virtual-env manager
- **[ruff](https://docs.astral.sh/ruff/)** — linter + formatter
- **[pyrefly](https://pyrefly.org/)** — static type checker
- **[pytest](https://pytest.org/)** — test framework

## Development

```bash
# Install with dev extras
uv sync --extra dev

# Lint + format
ruff check src tests
ruff format src tests

# Type check
pyrefly check src

# Tests
uv run pytest
```

See [AGENTS.md](AGENTS.md) for AI coding-agent guidelines.
