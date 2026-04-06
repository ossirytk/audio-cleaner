"""Separate a mixed audio file using a trained HDemucs checkpoint.

The "vocals" output contains the isolated jingle / advert signal.

Usage::

    uv run python -m scripts.separate_audio

Override the model folder or input/output paths in ``scripts/config.py`` or via
the ``JINGLE_BASE_DIR`` environment variable.
"""

import os
from pathlib import Path

import torch
import torchaudio

from scripts.config import INFERENCE_INPUT_FILE, INFERENCE_OUTPUT_DIR, MODEL_OUTPUTS_DIR


def find_latest_checkpoint(model_folder: str = "") -> Path:
    """Return the checkpoint path, either from a named folder or the most recently modified one."""
    if model_folder:
        return MODEL_OUTPUTS_DIR / model_folder / "checkpoint.th"

    if not MODEL_OUTPUTS_DIR.is_dir():
        msg = (
            f"Model outputs directory not found: {MODEL_OUTPUTS_DIR}. "
            "Run training first to generate experiment checkpoints."
        )
        raise FileNotFoundError(msg)

    all_xps = [d for d in MODEL_OUTPUTS_DIR.iterdir() if d.is_dir()]
    if not all_xps:
        msg = f"No experiment folders found under {MODEL_OUTPUTS_DIR}"
        raise FileNotFoundError(msg)
    latest_xp = max(all_xps, key=os.path.getmtime)
    print(f"Using latest experiment automatically: {latest_xp.name}")
    return latest_xp / "checkpoint.th"


def separate_audio(
    model_folder: str = "",
    input_file: Path = INFERENCE_INPUT_FILE,
    output_dir: Path = INFERENCE_OUTPUT_DIR,
) -> None:
    from demucs.apply import apply_model
    from demucs.hdemucs import HDemucs

    output_dir.mkdir(exist_ok=True, parents=True)

    model_path = find_latest_checkpoint(model_folder)
    if not model_path.exists():
        msg = f"Checkpoint not found: {model_path}"
        raise FileNotFoundError(msg)

    print(f"Loading checkpoint: {model_path}")
    package = torch.load(model_path, map_location="cpu", weights_only=False)

    if "kwargs" in package:
        model = HDemucs(**package["kwargs"])
    else:
        model = HDemucs(sources=["drums", "bass", "other", "vocals"])

    if "state" in package:
        model.load_state_dict(package["state"])
    elif "state_dict" in package:
        model.load_state_dict(package["state_dict"])
    else:
        msg = "Checkpoint contains neither 'state' nor 'state_dict'."
        raise KeyError(msg)

    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    print("HDemucs model loaded and set to eval mode.")

    print(f"Loading audio: {input_file}")
    wav, sr = torchaudio.load(str(input_file))

    if sr != 44100:
        wav = torchaudio.transforms.Resample(sr, 44100)(wav)

    ref = wav.abs().max()
    if ref > 1:
        wav = wav / ref

    print("Running separation... (this may take a moment)")
    with torch.no_grad():
        if torch.cuda.is_available():
            wav = wav.cuda()
        out = apply_model(model, wav[None], shifts=4, split=True, overlap=0.5)[0]

    sources = ["drums", "bass", "other", "vocals"]
    for i, name in enumerate(sources):
        save_path = output_dir / f"result_{name}.wav"
        torchaudio.save(str(save_path), out[i].cpu(), 44100)
        print(f"Saved: {save_path}")

    print(f"\nDone! Results in: {output_dir}")


def main() -> None:
    separate_audio()


if __name__ == "__main__":
    main()
