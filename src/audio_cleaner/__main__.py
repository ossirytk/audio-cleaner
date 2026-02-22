"""Command-line entry-point for audio-cleaner."""

from __future__ import annotations

import argparse
import sys

from audio_cleaner import __version__


def main() -> None:
    """Run the audio-cleaner CLI."""
    parser = argparse.ArgumentParser(
        prog="audio-cleaner",
        description="Clean FLAC and WAV audio files: denoise, normalise, remove ads.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # denoise sub-command (placeholder)
    denoise_parser = subparsers.add_parser("denoise", help="Remove background noise from audio files.")
    denoise_parser.add_argument("input", help="Input audio file or directory.")
    denoise_parser.add_argument("--output", "-o", default=".", help="Output directory (default: current dir).")

    # normalise sub-command (placeholder)
    normalise_parser = subparsers.add_parser("normalise", help="Improve audio quality (normalise, EQ, de-clip).")
    normalise_parser.add_argument("input", help="Input audio file or directory.")
    normalise_parser.add_argument("--output", "-o", default=".", help="Output directory (default: current dir).")

    # remove-ads sub-command (placeholder)
    ads_parser = subparsers.add_parser("remove-ads", help="Detect and remove ads / interrupts.")
    ads_parser.add_argument("input", help="Input audio file or directory.")
    ads_parser.add_argument("--output", "-o", default=".", help="Output directory (default: current dir).")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    print(f"Command '{args.command}' is not yet implemented. See docs/ for the implementation plan.")
    sys.exit(1)


if __name__ == "__main__":
    main()
