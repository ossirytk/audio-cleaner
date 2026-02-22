# Plan: Background Noise Removal

## Problem

Audio recordings often contain unwanted background noise that reduces intelligibility and listening quality:

- **Stationary noise** — constant broadband hiss, hum, electrical interference, fan noise, air conditioning.
- **Non-stationary noise** — intermittent sounds such as traffic, rain, crowd chatter, keyboard clicks.
- **Narrowband interference** — mains hum (50 Hz / 60 Hz and harmonics), fluorescent light buzz.
- **Room reverberation** — acoustic reflections that blur speech or music.

Supported input formats: **FLAC** and **WAV**.

---

## Proposed Methods

### 1. Spectral Subtraction (Stationary Noise)

**Goal:** Estimate and subtract the stationary noise power spectrum from the signal.

**Approach:**

1. Compute the Short-Time Fourier Transform (STFT) of the signal.
2. Estimate the noise power spectrum from a noise-only segment (first N frames assumed to be noise, or user-supplied noise profile).
3. Subtract the noise estimate from each STFT frame (with over-subtraction factor α and spectral floor β to reduce musical noise).
4. Reconstruct the signal via inverse STFT (ISTFT) with overlap-add.

**Python libraries:** `numpy`, `scipy.signal` (STFT/ISTFT), `scipy.fft`

**Notes:**
- Fast and effective for stationary noise.
- Produces "musical noise" artefacts at high over-subtraction factors; mitigate with a median smoothing of the noise estimate over time.

---

### 2. Wiener Filter (Stationary Noise)

**Goal:** Apply an optimal minimum mean-square error filter in the frequency domain.

**Approach:**

1. Estimate signal and noise PSDs (power spectral densities) from the STFT.
2. Compute the Wiener gain: `H(k) = SNR(k) / (1 + SNR(k))`.
3. Multiply each STFT frame by `H(k)` and reconstruct.

**Python libraries:** `numpy`, `scipy.signal`

**Notes:**
- Produces fewer artefacts than spectral subtraction.
- Can be combined with spectral subtraction (first pass) then refined with Wiener filtering.

---

### 3. `noisereduce` Library (Recommended Fast Path)

**Goal:** Provide a ready-to-use denoising solution for both stationary and non-stationary noise.

**Approach:**

1. Use the `noisereduce` Python library (pure Python, built on `scipy`/`numpy`).
2. For stationary noise: pass a noise clip or use the automatic noise estimation mode.
3. For non-stationary noise: use `prop_decrease` parameter and stationary=False mode.

**Python libraries:** `noisereduce`

**Notes:**
- `noisereduce` is the primary recommended approach because it is pure Python, actively maintained, and works well on speech and music.
- It uses a variant of spectral gating under the hood.
- No binary dependencies.

---

### 4. Notch Filter for Narrowband Hum

**Goal:** Remove mains hum (50 Hz or 60 Hz) and its harmonics.

**Approach:**

1. Detect the dominant low-frequency component using FFT (confirm whether 50 or 60 Hz is present).
2. Design a narrow IIR notch filter (2nd-order) at each harmonic (50, 100, 150 … Hz or 60, 120, 180 … Hz).
3. Apply all notch filters in sequence using `scipy.signal.sosfilt`.

**Python libraries:** `scipy.signal`, `numpy`

**Notes:**
- Very effective when hum is the dominant noise.
- Q factor of the notch filter is configurable (higher Q → narrower notch → less impact on nearby frequencies).

---

### 5. Voice Activity Detection (VAD) Gating

**Goal:** Zero out (or attenuate) frames where only background noise is present, preserving frames containing speech or music.

**Approach:**

1. Segment the signal into short frames (20–40 ms).
2. Compute energy and zero-crossing rate for each frame.
3. Classify frames as "active" or "silent/noise" using a threshold or a simple GMM-based VAD.
4. Attenuate noise frames by a configurable amount (soft gating) or zero them out (hard gating).

**Python libraries:** `numpy`, `scipy` (optionally `webrtcvad` for higher accuracy VAD — requires a C extension, treated as optional)

**Notes:**
- This is a gate, not a filter — it does not reconstruct signal buried in noise.
- Combine with spectral subtraction for best results: denoise first, then gate residual noise.

---

### 6. (Optional) Deep Learning Denoising

**Goal:** Use a neural network model for high-quality speech denoising.

**Approach:**

1. Use `speechbrain` or `denoiser` (Meta's demucs-based denoiser) for inference-only denoising.
2. Load a pre-trained model, run inference on overlapping windows, and reassemble.

**Python libraries:** `speechbrain` or `torch` + `denoiser`

**Notes:**
- Produces the best quality but requires large model files and `torch` (not pure Python).
- Treat as an **optional extra** (`pip install audio-cleaner[torch]`).
- Only use for speech; less appropriate for music.

---

## Pipeline Integration

```python
from audio_cleaner.noise import denoise

# Fast path — uses noisereduce
cleaned = denoise(audio_array, sample_rate, stationary=True)

# With a known noise profile clip
cleaned = denoise(audio_array, sample_rate, noise_clip=noise_array)
```

---

## Noise Profile Estimation

For recordings without a clean noise-only segment, two automatic strategies are offered:

1. **First-N-frames** — assume the first 500 ms of the file is noise (common for recordings that start before speech begins).
2. **Quietest-percentile** — identify the quietest 10 % of frames by energy and use their average spectrum as the noise profile.

---

## Acceptance Criteria

- [ ] Stationary noise removal via `noisereduce` (spectral gating).
- [ ] Narrowband hum removal via notch filters at 50/60 Hz and harmonics.
- [ ] Automatic noise profile estimation (first-N-frames and quietest-percentile modes).
- [ ] VAD-based soft gate as a post-processing step.
- [ ] Optional deep-learning denoising behind an extra install flag.
- [ ] All core methods covered by unit tests with synthetic noisy audio.

---

## References

- Boll, S. (1979). Suppression of acoustic noise in speech using spectral subtraction. *IEEE TASLP*.
- Wiener, N. (1949). Extrapolation, Interpolation, and Smoothing of Stationary Time Series.
- `noisereduce`: https://github.com/timsainburg/noisereduce
- `speechbrain`: https://speechbrain.github.io
- Meta `denoiser`: https://github.com/facebookresearch/denoiser
