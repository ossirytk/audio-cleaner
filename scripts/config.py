"""Central path configuration for all scripts.

Edit the constants below to match your local data layout, or override the base
directory via the ``JINGLE_BASE_DIR`` environment variable before running any script.

Expected directory layout::

    <BASE_DIR>/
    ├── music_sources/          # Raw FLAC music files (background content)
    ├── music_sources_cassettes/ # Optional second folder of raw FLAC music files
    ├── music_clips/            # 40-second WAV clips prepared from the sources above
    ├── jingles_original/       # Raw jingle / ident recordings (unprocessed)
    ├── jingles_processed/      # Normalised versions of the same jingles
    ├── training_dataset/
    │   ├── train/
    │   │   └── <track_name>/   # drums.wav  bass.wav  other.wav  vocals.wav  mixture.wav
    │   └── valid/
    ├── outputs/                # Training checkpoints (written by dora)
    └── test_audio/             # Input audio for the separation / cleaning step
"""

import os
from pathlib import Path

BASE_DIR = Path(os.environ.get("JINGLE_BASE_DIR", r"I:\jingle_removal"))

# 40-second music clips used as the background ("other") stem during dataset generation.
INPUT_MUSIC_DIR: Path = BASE_DIR / "music_clips"

# Jingle stems used for dataset construction
PROCESSED_JINGLES_DIR: Path = BASE_DIR / "jingles_processed"
ORIGINAL_JINGLES_DIR: Path = BASE_DIR / "jingles_original"

# Destination dataset (musdb-style structure)
OUTPUT_DATASET_DIR: Path = BASE_DIR / "training_dataset"

# Source folders containing raw FLAC files to slice into music clips
SOURCE_MUSIC_FOLDERS: list[Path] = [
    BASE_DIR / "music_sources",
    BASE_DIR / "music_sources_cassettes",
]

# Inference inputs / outputs
INFERENCE_INPUT_FILE: Path = BASE_DIR / "test_audio" / "mixture.wav"
INFERENCE_OUTPUT_DIR: Path = BASE_DIR / "separation_results"
MODEL_OUTPUTS_DIR: Path = BASE_DIR / "outputs" / "xps"
