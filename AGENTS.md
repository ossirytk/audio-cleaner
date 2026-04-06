# AGENTS.md

This file provides guidance for AI coding agents (GitHub Copilot, Cursor, etc.) working on the **audio-cleaner** project.

The repository is currently developed primarily from a **Windows dev drive** using **PowerShell** and **VS Code**.
WSL/Ubuntu and alternative editors remain supported as long as workflows stay terminal-reproducible.

## Available CLI Tools

| Tool | Purpose |
|------|---------|
| `rg` | Fast file content search (ripgrep) |
| `fd` | Fast file finder |
| `fzf` | Fuzzy finder for interactive selection |
| `ast-grep` | Structural code search and rewrite (AST-aware) |
| `tokei` | Count lines of code by language |
| `diff` / `diffutils` | File diffing |
| `zip` | Archive creation |
| `jq` | JSON query and transformation CLI |
| `yq` | YAML/JSON/TOML query and transformation CLI |
| `hyperfine` | Command-line benchmarking with statistical output |
| `pre-commit` | Run and manage repository pre-commit hooks |
| `http` / `https` (HTTPie) | Human-friendly HTTP API client |
| `difft` (difftastic) | Syntax-aware structural diffing |
| `bat` | Cat clone with syntax highlighting and Git integration |
| `delta` | Syntax-highlighting pager for git diffs (git-delta) |
| `sd` | Intuitive find-and-replace CLI (sed alternative) |
| `tldr` | Simplified, community-driven man pages (tealdeer) |
| `grex` | Generate regular expressions from example strings |

## Copilot Tooling Preferences

Use these defaults so Copilot has predictable, low-friction command choices.

### Preferred command order

- Content search: `rg` first, then `ast-grep` for structural/language-aware matching.
- File discovery: `fd` first, then `rg --files` as a fallback.
- JSON config inspection: `jq`.
- YAML/TOML inspection: `yq`.
- HTTP/API smoke checks: `http` / `https` (HTTPie).
- Diff/review: `difft` for syntax-aware diffs, `diff` for plain text diffs, `delta` for git diffs
- Performance comparisons: `hyperfine` for repeatable timing.

### Avoid in autonomous runs

- Avoid interactive-only flows (for example `fzf` prompts) unless explicitly requested.
- Avoid destructive git or file operations unless explicitly approved.
- Avoid long-running watch commands by default; use one-shot checks first.
- Avoid `pre-commit run --all-files` on very large repos when scoped checks are sufficient.

## Project Overview

`audio-cleaner` contains tooling to **train a custom HDemucs model** that detects and removes
radio jingles and adverts from audio recordings via source separation.  The model treats jingles
as the *vocals* stem and clean background music as the *other* stem.

## Repository Layout

```
audio-cleaner/
├── scripts/                 # HDemucs training utility scripts
│   ├── config.py            # Centralised path config (override with JINGLE_BASE_DIR)
│   ├── create_samples.py    # Slice source FLACs into 40-second WAV clips
│   ├── generate_dataset.py  # Build musdb-style training dataset
│   └── separate_audio.py    # Run inference with a trained HDemucs checkpoint
├── patches/                 # Overrides for installed demucs package files
│   ├── README.md            # Documents each patch and why it exists
│   └── demucs/              # Modified demucs library files (wav.py, train.py, …)
├── conf/                    # Hydra config for demucs training
│   └── config.yaml
├── justfile                 # Task runner recipes
├── pyproject.toml           # Project metadata and ruff config
├── AGENTS.md                # ← you are here
└── README.md
```

Data lives under `I:\jingle_removal\` by default (override with `JINGLE_BASE_DIR`):

```
I:\jingle_removal\
├── music_sources\             # Raw FLAC music files
├── music_sources_cassettes\   # Optional second raw FLAC folder
├── music_clips\               # 40-second WAV clips (create-samples output)
├── jingles_original\          # Raw jingle recordings
├── jingles_processed\         # Normalised jingle recordings
├── training_dataset\          # musdb-style dataset (generate-dataset output)
│   ├── train\<track>\  {drums,bass,other,vocals,mixture}.wav
│   └── valid\<track>\  …
├── outputs\                   # Dora training checkpoints
└── test_audio\mixture.wav     # Input for inference
```

## Toolchain

| Tool   | Purpose                       | Command                          |
|--------|-------------------------------|----------------------------------|
| `uv`   | Package & virtual-env manager | `uv sync` / `uv run`             |
| `ruff` | Linter + code formatter       | `uv run ruff check scripts`      |

### Setup

```powershell
# Install uv (if not already installed on Windows)
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install training extras + apply demucs/dora patches
uv sync --extra dev --extra training
uv run python scripts/apply_patches.py

# Override torch with a CUDA build (GPU training)
uv pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu126
uv run python scripts/apply_patches.py
```

### Running quality checks

```powershell
uv run ruff check scripts        # lint
uv run ruff format --check scripts  # format check
uv run ruff check --fix scripts  # auto-fix
```

## Coding Standards

- **Python ≥ 3.12** — use modern syntax (f-strings, `match`, `|` union types, etc.).
- **Type annotations** — every public function and method must have full type annotations.
- **Docstrings** — use Google-style docstrings for all public symbols.
- **Line length** — maximum 120 characters (enforced by ruff).
- **Imports** — stdlib → third-party → first-party, separated by blank lines (enforced by ruff/isort).
- **No bare `except`** — always catch specific exception types.

## PR Guidelines

- Keep PRs focused: one feature / fix per PR.
- Run `uv run ruff check scripts` before opening a PR.
- PR titles should follow Conventional Commits: `feat:`, `fix:`, `docs:`, `chore:`.
