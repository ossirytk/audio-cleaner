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
                timestamp_action=args.timestamp_action,
                timestamp_duck_db=args.timestamp_duck_db,
                cut_snap_ms=args.cut_snap_ms,
                cut_match_ms=args.cut_match_ms,
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


def _run_learn_ads(args: argparse.Namespace) -> None:
    """Execute the learn-ads sub-command.

    Args:
        args: Parsed CLI arguments.
    """
    import soundfile as sf

    from audio_cleaner.ads import create_ad_profile, save_ad_profile

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Error: '{input_path}' is not a valid file.", file=sys.stderr)
        sys.exit(1)
    if input_path.suffix.lower() not in {".flac", ".wav"}:
        print(
            f"Error: '{input_path}' is not a supported audio file. "
            "Supported extensions: .flac, .wav.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Parse timestamps: each entry is "start,end" in seconds
    if not args.timestamps:
        print(
            "Error: learn-ads requires at least one --timestamps interval.",
            file=sys.stderr,
        )
        sys.exit(1)

    timestamps: list[tuple[float, float]] = []
    for ts in args.timestamps:
        parts = ts.split(",")
        if len(parts) != 2:
            print(
                f"Error: invalid timestamp '{ts}'. "
                "Expected format: start,end (e.g. 102.0,106.0).",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            start_f = float(parts[0].strip())
            end_f = float(parts[1].strip())
        except ValueError:
            print(
                f"Error: non-numeric timestamp values in '{ts}'. "
                "Expected numeric seconds as start,end (e.g. 102.0,106.0).",
                file=sys.stderr,
            )
            sys.exit(1)
        timestamps.append((start_f, end_f))

    print(f"Loading: {input_path}")
    try:
        audio, sr = sf.read(str(input_path), dtype="float32")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: could not load '{input_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    if args.resample_hz and args.resample_hz != sr:
        from audio_cleaner.ads import _resample_audio  # type: ignore[attr-defined]

        print(f"  Resampling from {sr} Hz to {args.resample_hz} Hz …")
        audio = _resample_audio(audio, sr, args.resample_hz)  # type: ignore[arg-type]
        sr = args.resample_hz

    print(f"  Learning profile from {len(timestamps)} interval(s) …")
    profile = create_ad_profile(
        audio,  # type: ignore[arg-type]
        sr,
        timestamps,
        snap_ms=args.snap_ms,
        match_ms=args.match_ms,
        created_from=input_path.name,
    )

    if not profile.fingerprints:
        print(
            "Warning: no valid fingerprints were extracted. "
            "Check that the timestamp intervals are long enough (>= 0.5 s).",
            file=sys.stderr,
        )
        sys.exit(1)

    save_ad_profile(profile, args.profile_out)
    print(
        f"  Saved profile ({len(profile.fingerprints)} fingerprint(s)) → "
        f"{Path(args.profile_out).with_suffix('')}.json / .npz"
    )


def _run_apply_ads_profile(args: argparse.Namespace) -> None:
    """Execute the apply-ads-profile sub-command.

    Args:
        args: Parsed CLI arguments.
    """
    import soundfile as sf

    from audio_cleaner.ads import clean_with_profile, load_ad_profile

    profile_path = Path(args.profile)
    try:
        profile = load_ad_profile(str(profile_path))
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Loaded profile: {len(profile.fingerprints)} fingerprint(s), "
        f"sr={profile.sample_rate} Hz, source='{profile.created_from}'"
    )

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

    action: Literal["remove", "replace", "duck"] = args.action
    errors: list[str] = []

    for audio_file in audio_files:
        print(f"Processing: {audio_file}")
        try:
            audio, sr = sf.read(str(audio_file), dtype="float32")
            cleaned = clean_with_profile(
                audio,  # type: ignore[arg-type]
                sr,
                profile,
                action=action,
                fade_ms=args.fade_ms,
                duck_db=args.duck_db,
            )
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
        description="Clean FLAC and WAV audio files: normalise and remove ads.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

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
        default=60.0,
        dest="fade_ms",
        help="Crossfade duration at cut points in ms (default: 60.0).",
    )
    ads_parser.add_argument(
        "--timestamp-action",
        choices=("replace", "duck", "remove"),
        default="replace",
        dest="timestamp_action",
        help=(
            "How to process --timestamps intervals: 'replace' keeps duration with "
            "smooth bridging, 'duck' attenuates in-place, 'remove' cuts out "
            "(default: replace)."
        ),
    )
    ads_parser.add_argument(
        "--timestamp-duck-db",
        type=float,
        default=-18.0,
        dest="timestamp_duck_db",
        help="Gain in dB applied when --timestamp-action duck (default: -18.0).",
    )
    ads_parser.add_argument(
        "--cut-snap-ms",
        type=float,
        default=250.0,
        dest="cut_snap_ms",
        help=(
            "Search window in ms to snap remove-mode cut boundaries to nearby "
            "low-amplitude points (default: 250.0; set 0 to disable)."
        ),
    )
    ads_parser.add_argument(
        "--cut-match-ms",
        type=float,
        default=40.0,
        dest="cut_match_ms",
        help=(
            "Context window in ms used to align post-cut audio with pre-cut "
            "waveform when snapping remove boundaries (default: 40.0)."
        ),
    )

    # learn-ads sub-command
    learn_parser = subparsers.add_parser(
        "learn-ads",
        help="Learn an ad fingerprint profile from a source audio file and rough timestamps.",
    )
    learn_parser.add_argument("--input", required=True, help="Source FLAC or WAV audio file.")
    learn_parser.add_argument(
        "--timestamps",
        nargs="+",
        metavar="START,END",
        help=(
            "One or more rough ad intervals as 'start,end' in seconds "
            "(e.g. --timestamps 102,106 181,185 438,442 532,536)."
        ),
    )
    learn_parser.add_argument(
        "--profile-out",
        required=True,
        dest="profile_out",
        metavar="BASE_PATH",
        help=(
            "Base path for the output profile files (without extension). "
            "Two files are written: <BASE_PATH>.json and <BASE_PATH>.npz."
        ),
    )
    learn_parser.add_argument(
        "--snap-ms",
        type=float,
        default=250.0,
        dest="snap_ms",
        help="Boundary search window in ms for low-energy snapping (default: 250.0).",
    )
    learn_parser.add_argument(
        "--match-ms",
        type=float,
        default=40.0,
        dest="match_ms",
        help="Context window in ms for end-boundary alignment (default: 40.0).",
    )
    learn_parser.add_argument(
        "--resample-hz",
        type=int,
        default=None,
        dest="resample_hz",
        metavar="HZ",
        help=(
            "Resample the source audio to this sample rate before learning "
            "(e.g. 16000).  Useful for cross-format profiles."
        ),
    )

    # apply-ads-profile sub-command
    apply_parser = subparsers.add_parser(
        "apply-ads-profile",
        help="Detect and clean ad breaks in audio using a saved fingerprint profile.",
    )
    apply_parser.add_argument(
        "--input", required=True, help="Input audio file or directory of FLAC/WAV files."
    )
    apply_parser.add_argument(
        "--profile",
        required=True,
        metavar="BASE_PATH",
        help="Base path of the ad profile (without extension; see learn-ads --profile-out).",
    )
    apply_parser.add_argument(
        "--output", "-o", default=".", help="Output directory (default: current dir)."
    )
    apply_parser.add_argument(
        "--action",
        choices=("remove", "replace", "duck"),
        default="remove",
        help=(
            "How to handle detected ad segments: 'remove' cuts them out, "
            "'replace' inserts a smooth bridge, 'duck' attenuates in-place "
            "(default: remove)."
        ),
    )
    apply_parser.add_argument(
        "--fade-ms",
        type=float,
        default=60.0,
        dest="fade_ms",
        help="Crossfade/fade ramp duration in ms for 'remove' and 'duck' (default: 60.0).",
    )
    apply_parser.add_argument(
        "--duck-db",
        type=float,
        default=-18.0,
        dest="duck_db",
        help="Gain in dB applied when --action duck (default: -18.0).",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "remove-ads":
        _run_remove_ads(args)
        sys.exit(0)

    if args.command == "learn-ads":
        _run_learn_ads(args)
        sys.exit(0)

    if args.command == "apply-ads-profile":
        _run_apply_ads_profile(args)
        sys.exit(0)

    print(
        f"Command '{args.command}' is not yet implemented. See docs/ for the implementation plan."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
