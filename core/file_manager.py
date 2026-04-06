"""Helpers for inspecting the data directories and available checkpoints.

All paths are resolved relative to ``JINGLE_BASE_DIR`` (via ``scripts.config``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.config import (
    INFERENCE_OUTPUT_DIR,
    INPUT_MUSIC_DIR,
    MODEL_OUTPUTS_DIR,
    ORIGINAL_JINGLES_DIR,
    OUTPUT_DATASET_DIR,
    PROCESSED_JINGLES_DIR,
)

_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}


@dataclass
class FileEntry:
    """Metadata for a single file in a data directory."""

    name: str
    path: Path
    size_mb: float


@dataclass
class CheckpointEntry:
    """A training experiment checkpoint discovered under ``outputs/xps/``."""

    xp_name: str
    path: Path
    mtime: float

    @property
    def is_latest(self) -> bool:
        """Set externally after sorting; True for the most-recently modified checkpoint."""
        return self._is_latest

    @is_latest.setter
    def is_latest(self, value: bool) -> None:
        self._is_latest = value

    def __post_init__(self) -> None:
        self._is_latest = False


def _list_audio_files(directory: Path) -> list[FileEntry]:
    """Return audio files in *directory* (non-recursive)."""
    if not directory.is_dir():
        return []
    return [
        FileEntry(name=p.name, path=p, size_mb=round(p.stat().st_size / 1_048_576, 2))
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in _AUDIO_EXTS
    ]


def list_music_clips() -> list[FileEntry]:
    """List prepared 40-second music clips."""
    return _list_audio_files(INPUT_MUSIC_DIR)


def list_jingles_original() -> list[FileEntry]:
    """List raw jingle recordings."""
    return _list_audio_files(ORIGINAL_JINGLES_DIR)


def list_jingles_processed() -> list[FileEntry]:
    """List normalised jingle recordings."""
    return _list_audio_files(PROCESSED_JINGLES_DIR)


def list_dataset_tracks() -> dict[str, list[str]]:
    """Return train/valid track names from the dataset directory.

    Returns:
        Dict with keys ``"train"`` and ``"valid"``, each a sorted list of
        track folder names.
    """
    result: dict[str, list[str]] = {"train": [], "valid": []}
    for split in ("train", "valid"):
        split_dir = OUTPUT_DATASET_DIR / split
        if split_dir.is_dir():
            result[split] = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
    return result


def list_checkpoints() -> list[CheckpointEntry]:
    """Discover HDemucs checkpoints under ``outputs/xps/``.

    Returns a list sorted newest-first with ``is_latest`` set on the first entry.
    """
    if not MODEL_OUTPUTS_DIR.is_dir():
        return []

    entries = []
    for xp_dir in MODEL_OUTPUTS_DIR.iterdir():
        ckpt = xp_dir / "checkpoint.th"
        if ckpt.exists():
            entries.append(CheckpointEntry(xp_name=xp_dir.name, path=ckpt, mtime=ckpt.stat().st_mtime))

    entries.sort(key=lambda e: e.mtime, reverse=True)
    if entries:
        entries[0].is_latest = True
    return entries


def list_inference_results() -> list[FileEntry]:
    """List audio stems produced by the last inference run."""
    return _list_audio_files(INFERENCE_OUTPUT_DIR)


def dir_summary() -> dict[str, int]:
    """Return a quick count of items in each data directory for the dashboard."""
    return {
        "music_clips": len(list_music_clips()),
        "jingles_original": len(list_jingles_original()),
        "jingles_processed": len(list_jingles_processed()),
        "train_tracks": len(list_dataset_tracks()["train"]),
        "valid_tracks": len(list_dataset_tracks()["valid"]),
        "checkpoints": len(list_checkpoints()),
    }
