"""Command-line entry-point for audio-cleaner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

from audio_cleaner import __version__

_AD_STRATEGIES = ("silence", "loudness", "spectral", "combined")


def _run_remove_ads(args: argparse.Namespace) -> None:
    """Execute the remove-ads sub-command.

    Args:
        args: Parsed CLI arguments.
    """
    import soundfile as sf

    from audio_cleaner.ads import remove_ads

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        audio_files = sorted(
            p for p in input_path.rglob("*") if p.suffix.lower() in {".flac", ".wav"}
        )
    elif input_path.is_file():
        audio_files = [input_path]
    else:
        print(f"Error: '{input_path}' is not a valid file or directory.", file=sys.stderr)
        sys.exit(1)

    if not audio_files:
        print(f"No FLAC or WAV files found in '{input_path}'.", file=sys.stderr)
        sys.exit(1)

    strategy: Literal["silence", "loudness", "spectral", "combined"] = args.strategy

    for audio_file in audio_files:
        print(f"Processing: {audio_file}")
        audio, sr = sf.read(str(audio_file), dtype="float32")
        cleaned = remove_ads(
            audio,
            sr,
            strategy=strategy,
            silence_threshold_db=args.silence_threshold_db,
            silence_min_duration_s=args.silence_min_duration_s,
            silence_max_duration_s=args.silence_max_duration_s,
            loudness_jump_db=args.loudness_jump_db,
            loudness_min_duration_s=args.loudness_min_duration_s,
            loudness_max_duration_s=args.loudness_max_duration_s,
            spectral_distance_threshold=args.spectral_distance_threshold,
            fade_ms=args.fade_ms,
        )
        out_path = output_dir / audio_file.name
        sf.write(str(out_path), cleaned, sr)
        removed_s = (len(audio) - len(cleaned)) / sr
        print(f"  -> {out_path}  (removed ~{removed_s:.1f} s)")


def main() -> None:
    """Run the audio-cleaner CLI."""
    parser = argparse.ArgumentParser(
        prog="audio-cleaner",
        description="Clean FLAC and WAV audio files: denoise, normalise, remove ads.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # denoise sub-command (placeholder)
    denoise_parser = subparsers.add_parser(
        "denoise", help="Remove background noise from audio files."
    )
    denoise_parser.add_argument("input", help="Input audio file or directory.")
    denoise_parser.add_argument(
        "--output", "-o", default=".", help="Output directory (default: current dir)."
    )

    # normalise sub-command (placeholder)
    normalise_parser = subparsers.add_parser(
        "normalise", help="Improve audio quality (normalise, EQ, de-clip)."
    )
    normalise_parser.add_argument("input", help="Input audio file or directory.")
    normalise_parser.add_argument(
        "--output", "-o", default=".", help="Output directory (default: current dir)."
    )

    # remove-ads sub-command
    ads_parser = subparsers.add_parser("remove-ads", help="Detect and remove ads / interrupts.")
    ads_parser.add_argument("input", help="Input audio file or directory.")
    ads_parser.add_argument(
        "--output", "-o", default=".", help="Output directory (default: current dir)."
    )
    ads_parser.add_argument(
        "--strategy",
        choices=_AD_STRATEGIES,
        default="silence",
        help="Ad detection strategy (default: silence).",
    )
    ads_parser.add_argument(
        "--silence-threshold-db",
        type=float,
        default=-45.0,
        dest="silence_threshold_db",
        help="Silence threshold in dBFS (default: -45.0).",
    )
    ads_parser.add_argument(
        "--silence-min-duration",
        type=float,
        default=1.0,
        dest="silence_min_duration_s",
        help="Minimum silent gap to remove in seconds (default: 1.0).",
    )
    ads_parser.add_argument(
        "--silence-max-duration",
        type=float,
        default=30.0,
        dest="silence_max_duration_s",
        help="Maximum silent gap to remove in seconds (default: 30.0).",
    )
    ads_parser.add_argument(
        "--loudness-jump-db",
        type=float,
        default=8.0,
        dest="loudness_jump_db",
        help="Loudness jump threshold in dB (default: 8.0).",
    )
    ads_parser.add_argument(
        "--loudness-min-duration",
        type=float,
        default=5.0,
        dest="loudness_min_duration_s",
        help="Minimum loud segment to remove in seconds (default: 5.0).",
    )
    ads_parser.add_argument(
        "--loudness-max-duration",
        type=float,
        default=120.0,
        dest="loudness_max_duration_s",
        help="Maximum loud segment to remove in seconds (default: 120.0).",
    )
    ads_parser.add_argument(
        "--spectral-distance-threshold",
        type=float,
        default=0.15,
        dest="spectral_distance_threshold",
        help="Cosine distance threshold for spectral detector (default: 0.15).",
    )
    ads_parser.add_argument(
        "--fade-ms",
        type=float,
        default=20.0,
        dest="fade_ms",
        help="Crossfade duration at cut points in ms (default: 20.0).",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "remove-ads":
        _run_remove_ads(args)
        sys.exit(0)

    print(
        f"Command '{args.command}' is not yet implemented. See docs/ for the implementation plan."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
