# audio-cleaner

A Python toolkit for cleaning **FLAC** and **WAV** audio files.

## Features

| Feature | Module | Plan |
|---|---|---|
| Loudness normalisation, de-clipping, EQ, compression | `audio_cleaner.quality` | [docs/plan-audio-quality.md](docs/plan-audio-quality.md) |
| Ad / interrupt detection and removal | `audio_cleaner.ads` | Implemented |
| Fingerprint-based ad profile learning and application | `audio_cleaner.ads` | [docs/plan-fingerprint-cleaning.md](docs/plan-fingerprint-cleaning.md) |

## Quick Start

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run the CLI
uv run audio-cleaner --help
```

## Ad Removal Modes

The `remove-ads` command supports non-destructive replacement, optional ducking,
and hard cuts for timestamp-marked ad intervals.

```bash
# Default timestamp behavior: replace ad intervals with smooth bridges (preserves duration)
uv run audio-cleaner remove-ads input.wav --output cleaned/ \
	--strategy timestamps --timestamps 30.0,45.0

# Ducking mode: lower the ad interval by 24 dB
uv run audio-cleaner remove-ads input.wav --output cleaned/ \
	--strategy timestamps --timestamps 30.0,45.0 \
	--timestamp-action duck --timestamp-duck-db -24

# Hard removal: physically cut timestamp intervals out
uv run audio-cleaner remove-ads input.wav --output cleaned/ \
	--strategy timestamps --timestamps 30.0,45.0 \
	--timestamp-action remove

# Hard removal with seam-optimized defaults (recommended)
uv run audio-cleaner remove-ads input.wav --output cleaned/ \
	--strategy timestamps --timestamps 30.0,45.0 \
	--timestamp-action remove --fade-ms 60 --cut-snap-ms 250 --cut-match-ms 40

# Combined mode: replace timestamp intervals and remove fingerprint matches
uv run audio-cleaner remove-ads input.wav --output cleaned/ \
	--strategy combined --timestamps 30.0,45.0 \
	--reference-clips known_ad.wav
```

## Fingerprint-Based Ad Profile

Use the **profile workflow** when ad breaks are structurally repeated and you have
rough timestamps from at least one source recording.  The workflow has two steps:

### Step 1 — learn a profile

```bash
# Learn from four known ad break positions in a source FLAC file.
# Produces ad_profile.json + ad_profile.npz.
uv run audio-cleaner learn-ads \
    --input source.flac \
    --timestamps 102,106 181,185 438,442 532,536 \
    --profile-out ad_profile

# Optional: downsample to 16 kHz to speed up matching on long files
uv run audio-cleaner learn-ads \
    --input source.flac \
    --timestamps 102,106 181,185 438,442 532,536 \
    --profile-out ad_profile \
    --resample-hz 16000
```

### Step 2 — apply the profile

```bash
# Remove detected ad breaks from a single file (hard cut with crossfade)
uv run audio-cleaner apply-ads-profile \
    --input new_episode.flac \
    --profile ad_profile \
    --output cleaned/ \
    --action remove

# Apply to an entire directory of FLAC/WAV files
uv run audio-cleaner apply-ads-profile \
    --input recordings/ \
    --profile ad_profile \
    --output cleaned/ \
    --action remove

# Duck (attenuate) rather than cut
uv run audio-cleaner apply-ads-profile \
    --input new_episode.wav \
    --profile ad_profile \
    --output cleaned/ \
    --action duck --duck-db -24
```

### Profile format

Two files are written side-by-side:

| File | Contents |
|---|---|
| `<base>.json` | Human-readable metadata (version, sample rate, per-fingerprint thresholds) |
| `<base>.npz` | Compact numpy arrays (time-domain templates + spectral signatures) |

```json
{
  "profile_version": 1,
  "sample_rate": 44100,
  "created_from": "source.flac",
  "fingerprints": [
    {
      "id": "ad_01",
      "duration_s": 4.02,
      "ncc_threshold": 0.72,
      "spec_threshold": 0.65,
      "confidence": 1.0
    }
  ],
  "refinement": { "snap_ms": 250.0, "match_ms": 40.0 }
}
```

### Python API

```python
import soundfile as sf
from audio_cleaner.ads import create_ad_profile, save_ad_profile
from audio_cleaner.ads import load_ad_profile, clean_with_profile

# Learn
audio, sr = sf.read("source.flac", dtype="float32")
profile = create_ad_profile(
    audio, sr,
    rough_timestamps=[(102.0, 106.0), (181.0, 185.0), (438.0, 442.0), (532.0, 536.0)],
    created_from="source.flac",
)
save_ad_profile(profile, "ad_profile")

# Apply
profile = load_ad_profile("ad_profile")
audio, sr = sf.read("new_episode.flac", dtype="float32")
cleaned = clean_with_profile(audio, sr, profile, action="remove")
sf.write("new_episode_cleaned.flac", cleaned, sr)
```

## Toolchain

- **[uv](https://github.com/astral-sh/uv)** — fast package & virtual-env manager
- **[ruff](https://docs.astral.sh/ruff/)** — linter + formatter
- **[pyrefly](https://pyrefly.org/)** — static type checker
- **[pytest](https://pytest.org/)** — test framework

## Development

```bash
# Install with dev extras
uv sync --extra dev

# Lint + format
ruff check src tests
ruff format src tests

# Type check
pyrefly check src

# Tests
uv run pytest
```

See [AGENTS.md](AGENTS.md) for AI coding-agent guidelines.
