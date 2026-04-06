"""Create 40-second WAV samples from a collection of FLAC source files.

Reads FLAC files from the source folders defined in ``scripts/config.py``,
resamples to 44.1 kHz stereo, normalises peak to -6 dBFS, and saves numbered
WAV files to ``TARGET_FOLDER``.

Existing files are skipped (incremental — safe to re-run).

Usage::

    uv run python -m scripts.create_samples
    # or:
    just create-samples
"""

import random
import re
from pathlib import Path

import torchaudio
from tqdm import tqdm

from scripts.config import INPUT_MUSIC_DIR, SOURCE_MUSIC_FOLDERS

TARGET_FOLDER: Path = INPUT_MUSIC_DIR
SAMPLE_DURATION: int = 40  # seconds
SAMPLE_RATE: int = 44100
MAX_FILENAME_LEN: int = 80


def find_last_index() -> int:
    """Return the highest sequential number already present in the target folder."""
    wav_files = list(TARGET_FOLDER.glob("*.wav"))
    if not wav_files:
        return 0
    numbers = []
    for f in wav_files:
        match = re.match(r"(\d+)", f.name)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers) if numbers else 0


def sanitise_filename(name: str, max_len: int = MAX_FILENAME_LEN) -> str:
    """Remove characters forbidden in Windows filenames and truncate if necessary."""
    clean = re.sub(r'[\\/*?:"<>|]', "", name)
    if len(clean) > max_len:
        clean = clean[:max_len].strip()
    return clean


def _existing_stems(folder: Path) -> set[str]:
    """Return the set of sanitized stems already present in *folder*.

    Output filenames follow the pattern ``{counter:04d} - {clean_name}.wav``;
    this function extracts the ``clean_name`` portion of each existing file.
    """
    stems: set[str] = set()
    for f in folder.glob("*.wav"):
        parts = f.stem.split(" - ", 1)
        if len(parts) == 2:
            stems.add(parts[1])
    return stems


def create_samples() -> None:
    TARGET_FOLDER.mkdir(parents=True, exist_ok=True)

    last_index = find_last_index()
    print(f"Resuming from index: {last_index + 1}")

    flac_files: list[Path] = []
    for src in SOURCE_MUSIC_FOLDERS:
        if src.exists():
            flac_files.extend(src.rglob("*.flac"))

    print(f"Found {len(flac_files)} FLAC files in total.")
    counter = last_index
    new_samples = 0

    # Precompute existing sanitized stems for O(1) exact-match duplicate detection.
    existing_stems = _existing_stems(TARGET_FOLDER)

    for file_path in tqdm(flac_files):
        clean_name = sanitise_filename(file_path.stem)

        # Skip if a file with the same stem already exists (exact match)
        if clean_name in existing_stems:
            continue

        try:
            waveform, sample_rate = torchaudio.load(str(file_path))
            total_frames = waveform.shape[-1]

            duration_frames = int(SAMPLE_DURATION * sample_rate)

            if total_frames <= duration_frames:
                start_frame = 0
            else:
                max_start = total_frames - duration_frames
                start_frame = random.randint(0, max_start)

            wav = waveform[:, start_frame : start_frame + duration_frames]
            del waveform

            # Ensure stereo
            if wav.shape[0] == 1:
                wav = wav.repeat(2, 1)
            elif wav.shape[0] > 2:
                wav = wav[:2, :]

            # Resample to target sample rate if needed
            if sample_rate != SAMPLE_RATE:
                wav = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)(wav)

            # Normalise peak to -6 dBFS (0.5 linear)
            current_peak = wav.abs().max()
            if current_peak > 1e-6:
                wav = wav * (0.5 / current_peak)

            counter += 1
            new_samples += 1
            output_name = f"{counter:04d} - {clean_name}.wav"
            torchaudio.save(str(TARGET_FOLDER / output_name), wav, SAMPLE_RATE)

        except Exception as e:
            print(f"\nError processing {file_path.name}: {e}")

    print(f"\nDone! Added {new_samples} new samples.")
    print(f"Total files in folder: {counter}")


def main() -> None:
    create_samples()


if __name__ == "__main__":
    main()
