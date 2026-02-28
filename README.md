# audio-cleaner

A Python toolkit for cleaning **FLAC** and **WAV** audio files.

## Features

| Feature | Module | Plan |
|---|---|---|
| Loudness normalisation, de-clipping, EQ, compression | `audio_cleaner.quality` | [docs/plan-audio-quality.md](docs/plan-audio-quality.md) |
| Ad / interrupt detection and removal | `audio_cleaner.ads` | Implemented |

## Quick Start

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run the CLI
uv run audio-cleaner --help
```

## Ad Removal Modes

The `remove-ads` command supports non-destructive replacement, optional ducking,
and hard cuts for timestamp-marked ad intervals.

```bash
# Default timestamp behavior: replace ad intervals with smooth bridges (preserves duration)
uv run audio-cleaner remove-ads input.wav --output cleaned/ \
	--strategy timestamps --timestamps 30.0,45.0

# Ducking mode: lower the ad interval by 24 dB
uv run audio-cleaner remove-ads input.wav --output cleaned/ \
	--strategy timestamps --timestamps 30.0,45.0 \
	--timestamp-action duck --timestamp-duck-db -24

# Hard removal: physically cut timestamp intervals out
uv run audio-cleaner remove-ads input.wav --output cleaned/ \
	--strategy timestamps --timestamps 30.0,45.0 \
	--timestamp-action remove

# Hard removal with seam-optimized defaults (recommended)
uv run audio-cleaner remove-ads input.wav --output cleaned/ \
	--strategy timestamps --timestamps 30.0,45.0 \
	--timestamp-action remove --fade-ms 60 --cut-snap-ms 250 --cut-match-ms 40

# Combined mode: replace timestamp intervals and remove fingerprint matches
uv run audio-cleaner remove-ads input.wav --output cleaned/ \
	--strategy combined --timestamps 30.0,45.0 \
	--reference-clips known_ad.wav
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
