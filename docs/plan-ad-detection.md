# Plan: Ad / Interrupt Detection and Removal

## Problem

Audio recordings — particularly internet radio streams, podcasts, and recorded broadcasts —
contain unwanted interruptions:

- **Advertisements** — commercial breaks inserted at various points.
- **Station IDs / jingles** — short clips (1–5 s) identifying a broadcaster.
- **Sponsorship reads** — spoken ads embedded in the programme content.

Supported input formats: **FLAC** and **WAV**.

> **Scope note:** The audio files are relatively short and ads occur only a few times
> for durations of a few seconds each.  Silence-gap detection and loudness-change
> detection are **out of scope** for this implementation.

---

## Methods

### 1. Timestamp-Based Removal (user-specified)

**Goal:** Allow the user to explicitly mark ad/jingle/sponsorship intervals for removal
when the timestamps are already known.

**Approach:**

1. Accept a list of `(start_s, end_s)` pairs from the user.
2. Convert each pair to sample indices.
3. Apply a short raised-cosine fade-out / fade-in at each cut point.
4. Concatenate the kept segments.

**Python libraries:** `numpy`, `soundfile`

**Notes:**
- Zero false-positives because the user identifies the segments manually.
- Ideal for short recordings where the user has already listened through the file.

---

### 2. Audio Fingerprinting (Known Ad / Jingle Clips)

**Goal:** Detect and remove known advertisement, station ID, or jingle clips automatically
by matching audio fingerprints against the target recording.

**Approach:**

1. Accept one or more reference audio clips (the known ad / jingle / sponsorship read).
2. For each reference clip, compute a normalized cross-correlation (NCC) against the
   target audio.
3. Flag positions where the NCC exceeds a configurable threshold.
4. Cut out flagged regions, applying a short crossfade at each boundary.

**Python libraries:** `numpy`, `scipy.signal` (FFT-based correlate)

**Notes:**
- Highly accurate for exact or near-exact copies of known clips.
- NCC is computed efficiently via `scipy.signal.correlate` (FFT method).
- Works well for short clips (jingles, station IDs) embedded in longer recordings.

---

### 3. Segment Removal and Reassembly

**Goal:** After segments are identified by either method above, cleanly remove them.

**Approach:**

1. Collect the list of `(start, end)` sample indices from all active detectors.
2. Merge overlapping or adjacent intervals.
3. Apply a 20 ms raised-cosine fade-out / fade-in at each cut point.
4. Concatenate kept segments into the final output array.
5. Write to the output file using `soundfile`.

**Python libraries:** `numpy`, `soundfile`

---

## Pipeline Integration

```python
import soundfile as sf
from audio_cleaner.ads import remove_ads

audio, sr = sf.read("podcast.flac", dtype="float32")

# Remove known segments by user-provided timestamps
cleaned = remove_ads(
    audio, sr,
    strategy="timestamps",
    timestamps=[(30.0, 45.0), (120.0, 135.0)],
)

# Remove known ad clip wherever it appears in the recording
ad_clip, _ = sf.read("known_ad.flac", dtype="float32")
cleaned = remove_ads(
    audio, sr,
    strategy="fingerprint",
    reference_clips=[ad_clip],
    correlation_threshold=0.7,
)

# Apply both methods together
cleaned = remove_ads(
    audio, sr,
    strategy="combined",
    timestamps=[(30.0, 45.0)],
    reference_clips=[ad_clip],
)
```

---

## Acceptance Criteria

- [x] Timestamp-based removal with configurable fade.
- [x] Fingerprint-based detection via normalized cross-correlation against reference clips.
- [x] Support for advertisements, station IDs / jingles, and sponsorship reads.
- [x] Clean segment removal with raised-cosine crossfade at cut points.
- [x] Combined strategy applying both timestamp and fingerprint removal in one pass.
- [x] All methods covered by unit tests with synthetic audio.

---

## References

- Wang, A. (2003). An industrial-strength audio search algorithm. *ISMIR 2003*.
- `scipy.signal.correlate`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html

