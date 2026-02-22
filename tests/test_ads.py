"""Unit tests for audio_cleaner.ads — ad/interrupt detection and removal."""

from __future__ import annotations

import numpy as np
import pytest

from audio_cleaner.ads import (
    LoudnessChangeDetector,
    SilenceDetector,
    SpectralDissimilarityDetector,
    _apply_crossfade,
    _merge_segments,
    _rms_db,
    remove_ads,
)

SAMPLE_RATE = 16000  # 16 kHz — fast enough for tests
FULL_SCALE_SINE_RMS_DB = 20 * np.log10(1 / np.sqrt(2))  # ~= -3.01 dBFS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine(freq: float, duration_s: float, sr: int = SAMPLE_RATE, amp: float = 0.5) -> np.ndarray:
    """Generate a mono sine-wave burst."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(duration_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a silent segment."""
    return np.zeros(int(sr * duration_s), dtype=np.float32)


def _concat(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# _rms_db
# ---------------------------------------------------------------------------


def test_rms_db_silence() -> None:
    assert _rms_db(np.zeros(100, dtype=np.float32)) == pytest.approx(-120.0)


def test_rms_db_full_scale() -> None:
    # Full-scale sine has RMS of 1/sqrt(2) ~= -3 dBFS
    t = np.linspace(0, 1, 16000, endpoint=False)
    sig = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    assert _rms_db(sig) == pytest.approx(FULL_SCALE_SINE_RMS_DB, abs=0.1)


# ---------------------------------------------------------------------------
# _merge_segments
# ---------------------------------------------------------------------------


def test_merge_segments_empty() -> None:
    assert _merge_segments([]) == []


def test_merge_segments_no_overlap() -> None:
    segs = [(0, 100), (200, 300), (400, 500)]
    assert _merge_segments(segs) == segs


def test_merge_segments_overlapping() -> None:
    segs = [(0, 200), (100, 300), (400, 500)]
    assert _merge_segments(segs) == [(0, 300), (400, 500)]


def test_merge_segments_adjacent() -> None:
    segs = [(0, 100), (100, 200)]
    assert _merge_segments(segs) == [(0, 200)]


# ---------------------------------------------------------------------------
# _apply_crossfade
# ---------------------------------------------------------------------------


def test_apply_crossfade_no_segments() -> None:
    audio = _sine(440, 1.0)
    out = _apply_crossfade(audio, [], SAMPLE_RATE)
    np.testing.assert_array_equal(out, audio)


def test_apply_crossfade_removes_middle() -> None:
    """Remove a 1-second silent segment from the middle of a 3-second recording."""
    part_a = _sine(440, 1.0)
    silence = _silence(1.0)
    part_b = _sine(440, 1.0)
    audio = _concat(part_a, silence, part_b)
    remove = [(SAMPLE_RATE, 2 * SAMPLE_RATE)]
    out = _apply_crossfade(audio, remove, SAMPLE_RATE, fade_ms=10.0)
    # Output should be approximately 2 s long
    assert abs(len(out) - 2 * SAMPLE_RATE) < SAMPLE_RATE * 0.1


def test_apply_crossfade_stereo() -> None:
    """Cross-fade should work on 2-D (samples x channels) arrays."""
    mono = _sine(440, 2.0)
    stereo = np.stack([mono, mono], axis=1)
    remove = [(SAMPLE_RATE // 2, SAMPLE_RATE)]
    out = _apply_crossfade(stereo, remove, SAMPLE_RATE, fade_ms=10.0)
    assert out.ndim == 2
    assert out.shape[1] == 2


# ---------------------------------------------------------------------------
# SilenceDetector
# ---------------------------------------------------------------------------


def test_silence_detector_no_silence() -> None:
    audio = _sine(440, 3.0)
    det = SilenceDetector(threshold_db=-45.0, min_duration_s=1.0)
    assert det.detect(audio, SAMPLE_RATE) == []


def test_silence_detector_detects_gap() -> None:
    """A 2-second silent gap should be detected."""
    audio = _concat(_sine(440, 1.0), _silence(2.0), _sine(440, 1.0))
    det = SilenceDetector(threshold_db=-45.0, min_duration_s=1.0, max_duration_s=30.0)
    segs = det.detect(audio, SAMPLE_RATE)
    assert len(segs) == 1
    start, end = segs[0]
    # The gap starts at ~1 s and ends at ~3 s
    assert start == pytest.approx(SAMPLE_RATE, abs=SAMPLE_RATE * 0.05)
    assert end == pytest.approx(3 * SAMPLE_RATE, abs=SAMPLE_RATE * 0.05)


def test_silence_detector_ignores_short_gap() -> None:
    """A 0.5-second gap is shorter than min_duration_s and should not be flagged."""
    audio = _concat(_sine(440, 1.0), _silence(0.5), _sine(440, 1.0))
    det = SilenceDetector(threshold_db=-45.0, min_duration_s=1.0)
    assert det.detect(audio, SAMPLE_RATE) == []


def test_silence_detector_ignores_long_gap() -> None:
    """A 35-second gap is longer than max_duration_s=30 and should not be flagged."""
    audio = _concat(_sine(440, 1.0), _silence(35.0), _sine(440, 1.0))
    det = SilenceDetector(threshold_db=-45.0, min_duration_s=1.0, max_duration_s=30.0)
    assert det.detect(audio, SAMPLE_RATE) == []


def test_silence_detector_stereo() -> None:
    mono_gap = _concat(_sine(440, 1.0), _silence(2.0), _sine(440, 1.0))
    stereo = np.stack([mono_gap, mono_gap], axis=1)
    det = SilenceDetector(threshold_db=-45.0, min_duration_s=1.0)
    segs = det.detect(stereo, SAMPLE_RATE)
    assert len(segs) == 1


# ---------------------------------------------------------------------------
# LoudnessChangeDetector
# ---------------------------------------------------------------------------


def test_loudness_detector_no_change() -> None:
    """Uniform audio should produce no segments."""
    audio = _sine(440, 30.0, amp=0.1)
    det = LoudnessChangeDetector(loudness_jump_db=8.0, min_duration_s=5.0)
    assert det.detect(audio, SAMPLE_RATE) == []


def test_loudness_detector_detects_loud_burst() -> None:
    """A loud burst embedded in quiet audio should be detected."""
    quiet = _sine(440, 10.0, amp=0.01)
    loud = _sine(440, 15.0, amp=0.9)
    quiet2 = _sine(440, 10.0, amp=0.01)
    audio = _concat(quiet, loud, quiet2)
    det = LoudnessChangeDetector(loudness_jump_db=8.0, min_duration_s=5.0, max_duration_s=120.0)
    segs = det.detect(audio, SAMPLE_RATE)
    assert len(segs) >= 1
    # The detected segment should be roughly in the middle 10-25 s region
    start_s = segs[0][0] / SAMPLE_RATE
    assert 5.0 <= start_s <= 15.0


def test_loudness_detector_too_short() -> None:
    """A loud burst shorter than min_duration_s should not be returned."""
    quiet = _sine(440, 10.0, amp=0.01)
    loud = _sine(440, 2.0, amp=0.9)  # only 2 s < min_duration_s=5.0
    quiet2 = _sine(440, 10.0, amp=0.01)
    audio = _concat(quiet, loud, quiet2)
    det = LoudnessChangeDetector(loudness_jump_db=8.0, min_duration_s=5.0)
    assert det.detect(audio, SAMPLE_RATE) == []


# ---------------------------------------------------------------------------
# SpectralDissimilarityDetector
# ---------------------------------------------------------------------------


def test_spectral_detector_uniform_audio() -> None:
    """Uniform audio (same spectral content throughout) should produce no segments."""
    audio = _sine(440, 30.0, amp=0.3)
    det = SpectralDissimilarityDetector(
        distance_threshold=0.15,
        baseline_windows=5,
        min_duration_s=3.0,
        max_duration_s=30.0,
    )
    segs = det.detect(audio, SAMPLE_RATE)
    assert segs == []


def test_spectral_detector_detects_different_tone() -> None:
    """A long burst of spectrally different audio (white noise) should be flagged."""
    rng = np.random.default_rng(42)
    baseline = _sine(440, 10.0, amp=0.3)  # 440 Hz sine baseline
    # White noise has a very different spectral envelope than a pure tone
    burst = rng.standard_normal(10 * SAMPLE_RATE).astype(np.float32) * 0.3
    tail = _sine(440, 10.0, amp=0.3)
    audio = _concat(baseline, burst, tail)
    det = SpectralDissimilarityDetector(
        distance_threshold=0.05,
        baseline_windows=5,
        min_duration_s=3.0,
        max_duration_s=30.0,
    )
    segs = det.detect(audio, SAMPLE_RATE)
    assert len(segs) >= 1


# ---------------------------------------------------------------------------
# remove_ads — integration
# ---------------------------------------------------------------------------


def test_remove_ads_silence_strategy() -> None:
    """remove_ads with silence strategy should remove a silent gap."""
    audio = _concat(_sine(440, 1.0), _silence(2.0), _sine(440, 1.0))
    out = remove_ads(
        audio,
        SAMPLE_RATE,
        strategy="silence",
        silence_threshold_db=-45.0,
        silence_min_duration_s=1.0,
        silence_max_duration_s=30.0,
    )
    assert len(out) < len(audio)


def test_remove_ads_loudness_strategy() -> None:
    """remove_ads with loudness strategy should remove a loud burst."""
    quiet = _sine(440, 10.0, amp=0.01)
    loud = _sine(440, 15.0, amp=0.9)
    quiet2 = _sine(440, 10.0, amp=0.01)
    audio = _concat(quiet, loud, quiet2)
    out = remove_ads(
        audio,
        SAMPLE_RATE,
        strategy="loudness",
        loudness_jump_db=8.0,
        loudness_min_duration_s=5.0,
        loudness_max_duration_s=120.0,
    )
    assert len(out) < len(audio)


def test_remove_ads_combined_strategy() -> None:
    """combined strategy should handle both silence and loudness."""
    quiet = _sine(440, 5.0, amp=0.01)
    loud = _sine(440, 10.0, amp=0.9)
    gap = _silence(2.0)
    quiet2 = _sine(440, 5.0, amp=0.01)
    audio = _concat(quiet, loud, gap, quiet2)
    out = remove_ads(audio, SAMPLE_RATE, strategy="combined")
    assert len(out) < len(audio)


def test_remove_ads_no_ads() -> None:
    """When nothing is detected, the output should be identical to the input."""
    audio = _sine(440, 3.0, amp=0.3)
    out = remove_ads(audio, SAMPLE_RATE, strategy="silence")
    np.testing.assert_array_equal(out, audio)


def test_remove_ads_stereo_silence() -> None:
    """remove_ads should handle stereo input."""
    mono_gap = _concat(_sine(440, 1.0), _silence(2.0), _sine(440, 1.0))
    stereo = np.stack([mono_gap, mono_gap], axis=1)
    out = remove_ads(stereo, SAMPLE_RATE, strategy="silence")
    assert out.ndim == 2
    assert out.shape[0] < stereo.shape[0]
