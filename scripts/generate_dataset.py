"""Generate the jingle-removal training dataset.

Builds a musdb-style dataset where:
- ``other.wav``   = a 40-second music clip (background music)
- ``vocals.wav``  = synthesised jingle track (randomised pitch, gain, position)
- ``drums.wav`` / ``bass.wav`` = silence (unused stems)
- ``mixture.wav`` = other + vocals summed together

Run::

    uv run python -m scripts.generate_dataset
    # or via justfile:
    just generate-dataset
"""

import random

import numpy as np
from pydub import AudioSegment

from scripts.config import (
    INPUT_MUSIC_DIR,
    ORIGINAL_JINGLES_DIR,
    OUTPUT_DATASET_DIR,
    PROCESSED_JINGLES_DIR,
)

DURATION_MS = 40 * 1000  # 40 seconds
TRAIN_SPLIT = 0.8
TARGET_PEAK_DB = -6.0  # 0.5 linear peak


def change_pitch(sound: AudioSegment, octaves: float) -> AudioSegment:
    """Shift pitch by *octaves* without significantly changing tempo."""
    new_sample_rate = int(sound.frame_rate * (2.0**octaves))
    return sound._spawn(sound.raw_data, overrides={"frame_rate": new_sample_rate}).set_frame_rate(sound.frame_rate)


def random_position(total_duration_ms: int, jingle_duration_ms: int) -> int:
    """Return a uniformly random start position for a jingle within the clip."""
    max_pos = max(0, total_duration_ms - jingle_duration_ms)
    return int(random.uniform(0.0, 1.0) * max_pos)


def process_files() -> None:
    music_files = sorted(f for f in INPUT_MUSIC_DIR.iterdir() if f.suffix in {".wav", ".flac"})
    split_idx = int(len(music_files) * TRAIN_SPLIT)

    jingles_processed = [
        AudioSegment.from_file(f) for f in PROCESSED_JINGLES_DIR.iterdir() if f.suffix in {".wav", ".flac"}
    ]
    jingles_original = [f for f in ORIGINAL_JINGLES_DIR.iterdir() if f.suffix in {".wav", ".flac"}]

    for idx, music_file in enumerate(music_files):
        subset = "train" if idx < split_idx else "valid"
        folder_name = music_file.stem
        target_path = OUTPUT_DATASET_DIR / subset / folder_name
        target_path.mkdir(parents=True, exist_ok=True)

        # --- other.wav: the background music stem ---
        other = AudioSegment.from_file(music_file).set_channels(2).set_frame_rate(44100)
        other = other[:DURATION_MS]
        other.export(target_path / "other.wav", format="wav")

        # --- drums.wav / bass.wav: unused stems (silence) ---
        silence = AudioSegment.silent(duration=DURATION_MS, frame_rate=44100).set_channels(2)
        silence.export(target_path / "drums.wav", format="wav")
        silence.export(target_path / "bass.wav", format="wav")

        # --- vocals.wav: synthesised jingle track ---
        vocals = AudioSegment.silent(duration=DURATION_MS, frame_rate=44100).set_channels(2)

        # Pre-processed jingles — 7 random placements with uniform position
        for _ in range(7):
            jingle = random.choice(jingles_processed)
            pos = random_position(DURATION_MS, len(jingle))
            vocals = vocals.overlay(jingle, position=pos)

        # Original jingles — 3-5 placements with random pitch and gain
        for _ in range(random.randint(3, 5)):
            jingle_path = random.choice(jingles_original)
            jingle = AudioSegment.from_file(jingle_path)

            # Gain drawn from a triangular distribution biased towards louder values
            gain_factor = random.triangular(0.2, 1.0, 0.9)
            gain_db = 20 * np.log10(max(gain_factor, 0.0001))
            jingle = jingle + gain_db

            # Pitch shift ±10 % (~±0.15 octaves)
            pitch_shift = random.uniform(-0.15, 0.15)
            jingle = change_pitch(jingle, pitch_shift)

            pos = random_position(DURATION_MS, len(jingle))
            vocals = vocals.overlay(jingle, position=pos)

        # Limit peak to -6 dBFS
        current_peak = vocals.max_dBFS
        if current_peak > TARGET_PEAK_DB:
            vocals = vocals - (current_peak - TARGET_PEAK_DB)

        vocals.export(target_path / "vocals.wav", format="wav")

        # --- mixture.wav: mathematical sum of other + vocals ---
        mixture = other.overlay(vocals)
        mixture.export(target_path / "mixture.wav", format="wav")

        print(f"Done: {subset}/{folder_name}")


def main() -> None:
    process_files()


if __name__ == "__main__":
    main()
