# Plan: Ad / Interrupt Detection and Removal

## Problem

Audio recordings — particularly internet radio streams, podcasts, and recorded broadcasts — contain unwanted interruptions:

- **Advertisements** — commercial breaks of 15–120 seconds inserted at regular intervals.
- **Station IDs / jingles** — short clips (1–5 s) identifying a broadcaster.
- **Silence gaps** — dead air inserted at edit points or between segments.
- **Sponsorship reads** — spoken ads that are harder to detect than audio-clip ads.
- **Abrupt level changes** — loud ads following quiet programme content.

Supported input formats: **FLAC** and **WAV**.

---

## Proposed Methods

### 1. Silence / Gap Detection

**Goal:** Identify and remove gaps or dead-air segments that do not belong to the main content.

**Approach:**

1. Compute the RMS energy of each short frame (20 ms).
2. Mark frames below a configurable threshold (default: −45 dBFS) as silent.
3. Merge consecutive silent frames into silent segments.
4. Remove segments longer than a minimum duration (default: 1 s) and shorter than a maximum (default: 30 s, to avoid removing intentional long pauses).
5. Re-join the remaining audio segments.

**Python libraries:** `numpy`, `soundfile`

**Notes:**
- This is the simplest and most reliable method.
- Useful as a pre-processing step before more advanced ad detection.

---

### 2. Audio Fingerprinting (Known Ad Clips)

**Goal:** Detect and remove known advertisement or jingle clips by comparing audio fingerprints.

**Approach:**

1. Build a fingerprint database from a library of known ads/jingles using spectral landmark hashing (similar to Shazam's algorithm).
2. Compute fingerprints for overlapping windows of the target audio.
3. Look up fingerprints in the database; flag matching regions.
4. Cut out flagged regions and optionally apply a short crossfade to avoid clicks.

**Python libraries:** `numpy`, `scipy.signal` (spectrogram peak picking)

**Notes:**
- Highly accurate for exact or near-exact copies of known ads.
- Requires maintaining an ad fingerprint database (user-provided or auto-populated).
- Pure-Python implementation of spectral landmark hashing is feasible but non-trivial.
- The `dejavu` library (https://github.com/worldveil/dejavu) provides a ready implementation but requires a database backend; a simpler in-memory variant will be implemented first.

---

### 3. Volume / Loudness Change Detection

**Goal:** Detect ad breaks by identifying sudden changes in loudness level that are characteristic of ad insertion.

**Approach:**

1. Compute a running short-term loudness (RMS or LUFS) for every 1-second window.
2. Apply a change-point detection algorithm (e.g. PELT or a simple derivative threshold) to find abrupt loudness jumps.
3. Classify candidate segments around change points as ads if their average loudness is significantly higher than the baseline programme loudness.
4. Optionally combine with duration heuristics (ads are typically 15 s, 30 s, or 60 s).

**Python libraries:** `numpy`, `ruptures` (change-point detection, optional), or a simple derivative-based detector using `numpy.diff`.

**Notes:**
- `ruptures` is an optional dependency (pure Python, no binary requirements).
- A simple rolling-std threshold approach with `numpy` is sufficient for a first implementation.

---

### 4. Spectral Dissimilarity Detection

**Goal:** Detect ad segments whose spectral content differs significantly from the surrounding programme.

**Approach:**

1. Compute MFCC (Mel-Frequency Cepstral Coefficients) features for 1-second windows.
2. Build a moving baseline of "programme" MFCCs.
3. Compute the cosine distance between each window's MFCCs and the baseline.
4. Flag windows with distance above a threshold as potentially non-programme.
5. Merge flagged windows into candidate segments; discard segments outside expected ad durations.

**Python libraries:** `librosa` (MFCC), `numpy`, `scipy.spatial.distance`

**Notes:**
- More robust than loudness-only detection for speech-based ads.
- Works best when the main content has a consistent spectral profile (e.g. a single speaker).

---

### 5. Pattern-Based Detection (Recurring Segments)

**Goal:** Find ad segments that recur multiple times throughout a long recording (e.g. the same ad plays every 20 minutes in a radio stream).

**Approach:**

1. Divide the recording into 30-second chunks.
2. Compute a compact audio fingerprint (e.g. chroma features or averaged MFCC vectors) for each chunk.
3. Use cosine similarity to find groups of highly similar chunks.
4. Treat recurring dissimilar-to-baseline chunks as ad candidates.

**Python libraries:** `librosa`, `numpy`, `scipy.spatial.distance`

**Notes:**
- Requires a recording longer than ~1 hour to have enough recurrences.
- Low false-positive rate because programme content rarely repeats exactly.

---

### 6. Segment Removal and Reassembly

**Goal:** After ad segments are identified by any of the above methods, cleanly remove them from the audio.

**Approach:**

1. Collect the list of (start, end) sample indices to remove, sorted and non-overlapping.
2. Extract the kept segments (everything outside the ad intervals).
3. Apply a short (10–50 ms) raised-cosine fade-out / fade-in at each cut point to avoid clicks.
4. Concatenate kept segments into the final output array.
5. Write to the output file using `soundfile`.

**Python libraries:** `numpy`, `soundfile`

---

## Decision Tree for Method Selection

```
Is a fingerprint database available?
├── Yes → Use Audio Fingerprinting (Method 2) — highest precision
└── No
    ├── Are loudness changes prominent? → Loudness Change Detection (Method 3)
    ├── Is the recording long (> 1 hour)? → Pattern-Based Detection (Method 5)
    └── Default → Silence Detection (Method 1) + Spectral Dissimilarity (Method 4)
```

---

## Pipeline Integration

```python
from audio_cleaner.ads import remove_ads, SilenceDetector, LoudnessChangeDetector

# Simple silence-based removal
cleaned = remove_ads(audio_array, sample_rate, strategy="silence")

# Loudness + spectral combined
cleaned = remove_ads(
    audio_array,
    sample_rate,
    strategy="combined",
    silence_threshold_db=-45,
    loudness_jump_db=8,
)
```

---

## Acceptance Criteria

- [ ] Silence/gap detection with configurable threshold and duration limits.
- [ ] Loudness change-point detection using `numpy.diff` (no optional dependencies required).
- [ ] Spectral dissimilarity detection using MFCC features via `librosa`.
- [ ] Clean segment removal with crossfade at cut points.
- [ ] Optional audio fingerprinting for known ad clips.
- [ ] Optional pattern-based detection for recurring ad segments.
- [ ] All core methods covered by unit tests with synthetic audio containing inserted silence or tone bursts.

---

## References

- Wang, A. (2003). An industrial-strength audio search algorithm. *ISMIR 2003*.
- `ruptures` change-point detection: https://centre-borelli.github.io/ruptures-docs/
- `librosa` feature extraction: https://librosa.org/doc/latest/feature.html
- `dejavu` audio fingerprinting: https://github.com/worldveil/dejavu
