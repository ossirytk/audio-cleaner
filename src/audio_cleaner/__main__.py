"""Command-line entry-point for audio-cleaner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

from audio_cleaner import __version__

_AD_STRATEGIES = ("timestamps", "fingerprint", "combined")


def _run_remove_ads(args: argparse.Namespace) -> None:
    """Execute the remove-ads sub-command.

    Args:
        args: Parsed CLI arguments.
    """
    import soundfile as sf

    from audio_cleaner.ads import AudioArray, remove_ads

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        audio_files = sorted(
            p for p in input_path.rglob("*") if p.suffix.lower() in {".flac", ".wav"}
        )
        is_dir_input = True
    elif input_path.is_file():
        if input_path.suffix.lower() not in {".flac", ".wav"}:
            print(
                f"Error: '{input_path}' is not a supported audio file. "
                "Supported extensions: .flac, .wav.",
                file=sys.stderr,
            )
            sys.exit(1)
        audio_files = [input_path]
        is_dir_input = False
    else:
        print(f"Error: '{input_path}' is not a valid file or directory.", file=sys.stderr)
        sys.exit(1)

    if not audio_files:
        print(f"No FLAC or WAV files found in '{input_path}'.", file=sys.stderr)
        sys.exit(1)

    strategy: Literal["timestamps", "fingerprint", "combined"] = args.strategy

    # Parse timestamps: each entry is "start,end" in seconds
    timestamps: list[tuple[float, float]] | None = None
    if args.timestamps:
        timestamps = []
        for ts in args.timestamps:
            parts = ts.split(",")
            if len(parts) != 2:
                print(
                    f"Error: invalid timestamp '{ts}'. "
                    "Expected format: start,end (e.g. 30.0,45.0).",
                    file=sys.stderr,
                )
                sys.exit(1)
            start_str, end_str = parts
            try:
                start_f = float(start_str.strip())
                end_f = float(end_str.strip())
            except ValueError:
                print(
                    f"Error: non-numeric timestamp values in '{ts}'. "
                    "Expected numeric seconds as start,end (e.g. 30.0,45.0).",
                    file=sys.stderr,
                )
                sys.exit(1)
            timestamps.append((start_f, end_f))

    # Load reference clips for fingerprint strategy
    reference_clips: list[AudioArray] | None = None
    if args.reference_clips:
        reference_clips = []
        for clip_path in args.reference_clips:
            try:
                clip, _ = sf.read(clip_path, dtype="float32")
                reference_clips.append(clip)  # type: ignore[arg-type]
            except (OSError, RuntimeError, ValueError) as exc:
                print(f"Error: could not load reference clip '{clip_path}': {exc}", file=sys.stderr)
                sys.exit(1)

    # Validate that required inputs are provided for the selected strategy
    if strategy == "timestamps" and not timestamps:
        print(
            "Error: strategy 'timestamps' requires at least one --timestamps interval.",
            file=sys.stderr,
        )
        sys.exit(1)
    if strategy == "fingerprint" and not reference_clips:
        print(
            "Error: strategy 'fingerprint' requires at least one --reference-clips file.",
            file=sys.stderr,
        )
        sys.exit(1)

    errors: list[str] = []
    for audio_file in audio_files:
        print(f"Processing: {audio_file}")
        try:
            audio, sr = sf.read(str(audio_file), dtype="float32")
            cleaned = remove_ads(
                audio,  # type: ignore[arg-type]
                sr,
                strategy=strategy,
                timestamps=timestamps,
                reference_clips=reference_clips,
                correlation_threshold=args.correlation_threshold,
                fade_ms=args.fade_ms,
            )
            # Preserve relative directory structure when input is a directory
            if is_dir_input:
                out_path = output_dir / audio_file.relative_to(input_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                out_path = output_dir / audio_file.name
            sf.write(str(out_path), cleaned, sr)
            removed_s = (len(audio) - len(cleaned)) / sr
            print(f"  -> {out_path}  (removed ~{removed_s:.1f} s)")
        except Exception as exc:
            msg = f"  ERROR: {audio_file}: {exc}"
            print(msg, file=sys.stderr)
            errors.append(msg)

    if errors:
        n = len(errors)
        noun = "file" if n == 1 else "files"
        print(f"\n{n} {noun} failed to process.", file=sys.stderr)
        sys.exit(1)


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
        default="timestamps",
        help="Ad detection strategy (default: timestamps).",
    )
    ads_parser.add_argument(
        "--timestamps",
        nargs="+",
        metavar="START,END",
        dest="timestamps",
        help=(
            "One or more time intervals to remove, each as 'start,end' in seconds "
            "(e.g. --timestamps 30.0,45.0 120.5,135.0)."
        ),
    )
    ads_parser.add_argument(
        "--reference-clips",
        nargs="+",
        metavar="FILE",
        dest="reference_clips",
        help=(
            "One or more FLAC/WAV reference clip files (ads, jingles, sponsorship reads) "
            "used for fingerprint-based detection."
        ),
    )
    ads_parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.7,
        dest="correlation_threshold",
        help=(
            "Minimum normalized cross-correlation score to flag a fingerprint match (default: 0.7)."
        ),
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
