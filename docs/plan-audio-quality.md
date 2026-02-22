# Plan: Audio Quality Improvement

## Problem

Audio recordings may suffer from various quality issues:

- **Low volume / under-normalised signal** — the RMS or peak level is much lower than the full digital range.
- **Clipping / over-driven signal** — samples exceed 0 dBFS, causing hard distortion.
- **Frequency imbalance** — excessive bass, muffled highs, or uneven spectral response.
- **Dynamic range issues** — large volume swings between quiet passages and loud peaks.
- **Sample-rate / bit-depth mismatch** — files encoded at non-standard rates that introduce aliasing.

Supported input formats: **FLAC** and **WAV**.

---

## Proposed Methods

### 1. Loudness Normalisation

**Goal:** Bring the perceived loudness to a target level (e.g. −14 LUFS for streaming, −23 LUFS for broadcast EBU R128).

**Approach:**

1. Measure integrated loudness using the ITU-R BS.1770 algorithm (available in `pyloudnorm`).
2. Calculate the required gain offset.
3. Apply linear gain and clip to −1.0 / +1.0.

**Python libraries:** `pyloudnorm`, `numpy`, `soundfile`

**Notes:**
- Also support peak normalisation (simpler: divide by `max(abs(signal))`).
- Offer a "true peak" mode to avoid inter-sample clipping after normalisation.

---

### 2. De-clipping

**Goal:** Reconstruct samples that were hard-clipped (saturated at ±1.0 or ±32767 for 16-bit PCM).

**Approach:**

1. Detect clipped regions: runs of consecutive samples at the maximum amplitude.
2. Use cubic-spline or sinc interpolation to estimate the true waveform over the clipped region.
3. Optionally apply a soft-knee limiter after reconstruction to prevent re-clipping.

**Python libraries:** `scipy.interpolate`, `numpy`

**Notes:**
- Works best when the clipping is short (< 5 ms). Heavily clipped material may need a different strategy.
- Flag files with > 0.1 % clipped samples for manual review.

---

### 3. Equalisation (EQ)

**Goal:** Correct tonal imbalances in the frequency response.

**Approach:**

1. **High-pass filter** — remove sub-bass rumble (< 80 Hz) that adds no musical content.
2. **Low-shelf boost** — compensate for thin-sounding recordings (optional, user-configurable).
3. **High-shelf boost** — restore air/presence in muffled recordings (optional).
4. **Notch filter** — remove narrow resonances (e.g. mains hum at 50/60 Hz).

**Python libraries:** `scipy.signal` (IIR filters: `butter`, `sosfilt`)

**Parameter exposure:** All filter frequencies and gains configurable via `pyproject.toml` `[tool.audio-cleaner.quality]` or CLI flags.

---

### 4. Dynamic Range Compression

**Goal:** Reduce the gap between loud and quiet parts so the audio is comfortable to listen to.

**Approach:**

1. Compute a short-time RMS envelope.
2. Apply a feed-forward compressor with configurable threshold, ratio, attack, and release.
3. Apply make-up gain to restore average loudness after compression.

**Python libraries:** `numpy`, `scipy`

**Notes:**
- This is a pure DSP operation; no external binary required.
- Expose ratio (e.g. 4:1), threshold (dBFS), attack (ms), and release (ms) as parameters.

---

### 5. Resampling / Format Standardisation

**Goal:** Convert non-standard sample rates (e.g. 22 050 Hz, 48 000 Hz) to a canonical rate (44 100 Hz or 48 000 Hz).

**Approach:**

1. Read the file with `soundfile`.
2. If the sample rate differs from the target, resample with `scipy.signal.resample_poly` (integer-ratio) or `librosa.resample` (arbitrary ratio).
3. Write the output back with `soundfile`.

**Python libraries:** `soundfile`, `scipy.signal`, `librosa`

---

## Pipeline Integration

The quality-improvement steps will be exposed as a composable pipeline:

```python
from audio_cleaner.quality import Pipeline, highpass_filter, normalise, decompress

pipeline = Pipeline([
    highpass_filter(cutoff_hz=80),
    normalise(target_lufs=-14),
])
cleaned = pipeline.run(audio_array, sample_rate)
```

Each step is a pure function `(np.ndarray, int) -> np.ndarray` to keep the pipeline stateless and testable.

---

## Acceptance Criteria

- [ ] Loudness normalisation to a configurable LUFS target.
- [ ] Peak normalisation fallback (no `pyloudnorm` dependency required for basic use).
- [ ] De-clipping for short clips (< 5 ms).
- [ ] Configurable high-pass filter.
- [ ] Dynamic range compressor with at least threshold and ratio parameters.
- [ ] Optional resampling to a canonical sample rate.
- [ ] All steps covered by unit tests with synthetic audio arrays.

---

## References

- ITU-R BS.1770-4 loudness measurement standard
- EBU R128 broadcast loudness recommendation
- `pyloudnorm`: https://github.com/csteinmetz1/pyloudnorm
- `scipy.signal` filter design: https://docs.scipy.org/doc/scipy/reference/signal.html
