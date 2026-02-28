# Plan: Fingerprint-Driven Ad Cleaning from Raw Audio

## Problem

In this project, ad breaks are:

- **Structurally repeated** (same content each occurrence)
- **At known approximate timestamps** in a source recording
- **Present in multiple encodings** of the same programme (e.g. FLAC and WAV)

Manual timestamp cutting works, but it is fragile when boundaries drift by a few hundred milliseconds or when new files have small timing offsets.

> **Data constraint:** In current real examples there are only **four ad breaks** per file.
> The design must work reliably in a sparse-data / few-shot setting.

---

## Goal

Build a feature that:

1. Learns one or more ad fingerprints directly from raw audio using known rough timestamps.
2. Stores those fingerprints as a reusable profile.
3. Applies the profile to clean new files automatically with minimal manual timestamp tuning.

---

## Scope

### In scope

- Fingerprint creation from a raw audio file + rough `(start, end)` ad intervals.
- Fingerprint matching against target audio to detect repeated ad breaks.
- Segment cleanup using existing remove/replace/duck actions.
- Profile persistence (`.json` metadata + compact numeric arrays).
- CLI workflow for `learn` then `apply`.
- Few-shot robustness with as few as 3–4 labeled ad intervals.

### Out of scope

- ML model training.
- Speaker/content semantic classification.
- Real-time streaming detection.

---

## Sparse-Data Requirements (Few-Shot)

Because only four ad examples may be available:

1. Do not rely on large-sample statistics or model training.
2. Prioritize deterministic DSP features (NCC + compact spectral signatures).
3. Keep thresholds adaptive to per-profile evidence (not hard global constants only).
4. Track confidence per detection and support conservative filtering.
5. Support optional fallback to user timestamps when confidence is low.

---

## Proposed Workflow

### Phase 1: Learn fingerprint profile from raw audio

Input:

- Source file (`.flac` or `.wav`)
- Rough known intervals (e.g. `102-106`, `181-185`, `438-442`, `532-536`)

Steps:

1. Load mono reference signal (`float32`).
2. For each rough interval, run boundary refinement:
   - low-energy/zero-crossing snap
   - optional local alignment to nearest repeated pattern
3. Extract cleaned clip candidates.
4. Normalize candidates (RMS normalization + optional band-pass).
5. Build fingerprint features per clip:
   - time-domain normalized template
   - lightweight spectral signature (e.g. log-mel or STFT peak hashes)
6. Validate consistency across clips (pairwise correlation threshold).
7. Save profile with metadata and per-clip fingerprints.

Sparse-data specifics:

- Build one fingerprint per interval first (no averaging required).
- Optionally build one consensus fingerprint only if pairwise similarity is high.
- Store per-fingerprint confidence so low-quality examples can be down-weighted.

Output:

- `ad_profile.json`
- Optional `ad_profile.npz` for numeric feature arrays

### Phase 2: Apply profile to target audio

Input:

- Target file(s)
- Saved ad profile

Steps:

1. Load target audio and profile.
2. For each stored fingerprint, compute candidate matches:
   - NCC in time domain
   - optional spectral score for confirmation
3. Fuse scores and keep matches above threshold.
4. Merge overlapping detections.
5. Clean detected segments using selected action:
   - `remove` (with fade + snap + match refinement)
   - `replace`
   - `duck`
6. Write cleaned output.

Sparse-data specifics:

- Use multi-template matching (all learned fingerprints vote).
- Accept a detection if either:
   - one strong template match is present, or
   - two moderate matches agree in location.
- Emit detection confidence report for optional manual review.

---

## Data Model

```json
{
  "profile_version": 1,
  "sample_rate": 16000,
  "created_from": "1992-06-16d1t01.flac",
  "fingerprints": [
    {
      "id": "ad_01",
      "duration_s": 4.02,
      "template_norm": "npz:template_01",
      "spectral_signature": "npz:spec_01",
      "ncc_threshold": 0.72,
      "spec_threshold": 0.65
    }
  ],
  "refinement": {
    "snap_ms": 250,
    "match_ms": 40
  }
}
```

---

## API Design

Add new functions in `audio_cleaner.ads`:

- `create_ad_profile(audio, sample_rate, rough_timestamps, *, snap_ms=250, match_ms=40) -> AdProfile`
- `save_ad_profile(profile, path) -> None`
- `load_ad_profile(path) -> AdProfile`
- `clean_with_profile(audio, sample_rate, profile, *, action="remove", fade_ms=60) -> np.ndarray`

---

## CLI Design

Add two subcommands:

1. `audio-cleaner learn-ads`
   - `--input`
   - `--timestamps`
   - `--profile-out`
   - optional: `--snap-ms`, `--match-ms`, `--resample-hz`

2. `audio-cleaner apply-ads-profile`
   - `--input` (file or directory)
   - `--profile`
   - `--output`
   - `--action {remove,replace,duck}`
   - optional: `--fade-ms`, `--cut-snap-ms`, `--cut-match-ms`

---

## Matching Strategy Details

### Primary detector: normalized cross-correlation (NCC)

- Fast, explainable, works well for repeated identical ad breaks.
- Use FFT-based correlation for speed.

### Secondary confirmation: spectral similarity

- Compute compact spectral vectors for candidate region and reference clip.
- Reject false positives where NCC is high but spectrum differs.

### Score fusion

`final_score = w_time * ncc + w_spec * spectral_score`

Default: `w_time=0.7`, `w_spec=0.3`.

### Sparse-data thresholding

- Initialize thresholds from profile-internal similarities (few-shot calibration).
- Use conservative floor thresholds to avoid false positives.
- Expose profile-level overrides for difficult recordings.

---

## Robustness Rules

- Skip fingerprints shorter than minimum duration (e.g. 0.5 s).
- De-duplicate near-identical fingerprints from multiple intervals.
- Clamp detections within audio bounds.
- Merge detections if overlap or if gap < 100 ms.
- Require minimum confidence for auto-clean; optionally emit low-confidence report.
- In few-shot mode, keep top-K non-overlapping candidates per fingerprint before merge.
- If detections are fewer than expected, return warnings instead of aggressive over-matching.

---

## Testing Plan

1. **Profile creation tests**
   - synthetic repeated ad inserted at known offsets
   - verify generated profile count and durations
2. **Detection tests**
   - exact copy in target audio -> high recall
   - no ad present -> low false positive rate
3. **Cross-format tests**
   - learn from FLAC, apply to WAV version of same content
4. **Boundary tolerance tests**
   - apply random ±300 ms drift to insertion points
5. **End-to-end tests**
   - `learn-ads` then `apply-ads-profile` CLI flow
6. **Sparse-data tests (4 examples)**
   - learn from exactly four ad intervals
   - verify stable matching on FLAC and WAV variants
   - verify confidence output and low false positives

---

## Acceptance Criteria

- [ ] Can create a reusable ad profile from raw source audio and rough timestamps.
- [ ] Profile can be applied to both FLAC and WAV versions of the same programme.
- [ ] Detection remains stable with at least ±300 ms timing drift.
- [ ] Cleaning action selectable (`remove`, `replace`, `duck`).
- [ ] End-to-end CLI flow works on file and directory inputs.
- [ ] Unit/integration tests cover profile generation, matching, and cleaning.
- [ ] Works with only four labeled ad breaks in the source file.
- [ ] Sparse-data mode maintains high precision (no obvious over-matching).

---

## Implementation Priority (4-Example Profiles)

1. **MVP profile learning (must-have)**
   - Implement `create_ad_profile` with one template per provided interval.
   - Reuse existing boundary refinement (`snap` + `match`) before extraction.
2. **Safe matching core (must-have)**
   - Implement NCC-based multi-template detection with conservative thresholds.
   - Keep top-K non-overlapping candidates per template.
3. **Confidence + guardrails (must-have)**
   - Emit per-detection confidence scores.
   - Warn when expected number of detections is not met.
4. **Cross-format validation (high priority)**
   - Learn on FLAC, apply on WAV (and vice versa) for the same programme.
5. **Spectral confirmation (high priority)**
   - Add lightweight spectral second-pass filtering to reduce false positives.
6. **CLI productization (follow-up)**
   - Add `learn-ads` and `apply-ads-profile` commands.
   - Include profile export/import and dry-run confidence reporting.

Deliver in this order to maximize reliability under sparse-data constraints before adding extras.

---

## Rollout Plan

1. Implement profile data structures + save/load.
2. Implement `create_ad_profile` with boundary refinement reuse.
3. Implement profile-based detection + cleaning API.
4. Add CLI subcommands.
5. Add tests and example commands in README.
6. Validate on existing known file pair (`1992-06-16d1t01.flac/.wav`).

---

## Definition of Done

The feature is complete when all items below are satisfied:

1. **Profile learning implemented**
   - `create_ad_profile`, `save_ad_profile`, and `load_ad_profile` are available.
   - Works from rough timestamps on a raw source file with only four ad examples.
2. **Profile application implemented**
   - `clean_with_profile` detects and cleans segments using a saved profile.
   - Supports actions: `remove`, `replace`, and `duck`.
3. **CLI workflow complete**
   - `learn-ads` and `apply-ads-profile` commands are implemented and documented.
   - Both file and directory inputs are supported.
4. **Sparse-data behavior verified**
   - Detection is stable using exactly four labeled intervals.
   - Confidence output/warnings are present for low-confidence or low-count detections.
5. **Cross-format verified**
   - Learn on FLAC and apply to WAV variant (and/or inverse) with expected detections.
6. **Quality gates pass**
   - Unit/integration tests for learning + matching + cleaning are in place and passing.
   - `ruff check src tests` passes.
7. **User-facing docs updated**
   - README includes minimal examples for learning a profile and applying it.
   - Profile format/options are explained at a practical level.
