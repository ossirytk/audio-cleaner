# audio-cleaner — justfile
# Run `just` to list all available recipes.
# Requires: just (https://github.com/casey/just), uv

set shell := ["pwsh", "-NoLogo", "-Command"]

default:
    @just --list

# Install dependencies (dev tools only)
sync:
    uv sync --extra dev

# Install including ML training dependencies
sync-training:
    uv sync --extra dev --extra training
    just apply-patches

# Copy patched demucs/dora files into the active .venv after uv sync.
# Run this once after `just sync-training`, and again whenever patches/ changes.
apply-patches:
    uv run python scripts/apply_patches.py

# Run HDemucs training via dora. Pass extra dora args after --:
#   just train dset.musdb=D:/data/my_dataset epochs=200
train *ARGS:
    uv run dora run model=hdemucs dset=musdb44 \
        epochs=100 \
        ++batch_size=2 \
        ++segments=8 \
        ++misc.num_workers=0 \
        ++dset.num_workers=0 \
        ++optim.lr=0.0002 \
        "++weights=[0.1,0.1,1.0,10.0]" \
        ++augment.remix.group_size=2 \
        +name=HIFI_HDEMUCS \
        ++test.metrics=False \
        ++test.every=200 \
        {{ARGS}}

# Create 40-second WAV samples from source FLAC files (training data prep)
create-samples:
    uv run python -m scripts.create_samples

# Build the musdb-style training dataset from samples + jingle stems
generate-dataset:
    uv run python -m scripts.generate_dataset

# Separate audio using the trained HDemucs model
separate:
    uv run python -m scripts.separate_audio

# --- Quality checks ---

lint:
    uv run ruff check scripts patches
    uv run ruff format --check scripts patches

fix:
    uv run ruff check --fix scripts patches
    uv run ruff format scripts patches

check: lint
