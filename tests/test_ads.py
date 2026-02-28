"""Unit tests for audio_cleaner.ads — ad/interrupt detection and removal."""

from __future__ import annotations

import numpy as np
import pytest

from audio_cleaner.ads import (
    FingerprintDetector,
    _apply_crossfade,
    _apply_ducking,
    _apply_replacement,
    _merge_segments,
    _normalized_cross_correlation,
    _snap_remove_boundaries,
    remove_ads,
)

SAMPLE_RATE = 16000  # 16 kHz — fast enough for tests


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


def test_apply_crossfade_overlap_join_avoids_deep_dip() -> None:
    """Crossfade join should not dip close to silence for equal-level neighboring chunks."""
    part_a = np.ones(SAMPLE_RATE, dtype=np.float32)
    removed_mid = np.zeros(SAMPLE_RATE, dtype=np.float32)
    part_b = np.ones(SAMPLE_RATE, dtype=np.float32)
    audio = _concat(part_a, removed_mid, part_b)
    out = _apply_crossfade(audio, [(SAMPLE_RATE, 2 * SAMPLE_RATE)], SAMPLE_RATE, fade_ms=50.0)

    join = SAMPLE_RATE - int(0.05 * SAMPLE_RATE)
    join_window = out[max(0, join - 20) : min(len(out), join + 20)]
    assert float(np.min(join_window)) > 0.2


def test_apply_ducking_preserves_length_and_reduces_level() -> None:
    """Ducking should attenuate the marked interval without changing duration."""
    audio = _sine(440, 3.0, amp=0.8)
    start = int(1.0 * SAMPLE_RATE)
    end = int(2.0 * SAMPLE_RATE)
    out = _apply_ducking(
        audio,
        duck_segments=[(start, end)],
        sample_rate=SAMPLE_RATE,
        duck_db=-18.0,
        fade_ms=20.0,
    )
    assert len(out) == len(audio)
    in_rms = np.sqrt(np.mean(audio[start:end] ** 2))
    out_rms = np.sqrt(np.mean(out[start:end] ** 2))
    assert out_rms < in_rms * 0.2


def test_apply_replacement_preserves_length_and_changes_segment() -> None:
    """Replacement should keep duration while replacing content in marked interval."""
    jingle = _sine(2000, 1.0)
    audio = _concat(_sine(440, 1.0), jingle, _sine(440, 1.0))
    start = SAMPLE_RATE
    end = 2 * SAMPLE_RATE
    out = _apply_replacement(audio, [(start, end)])
    assert len(out) == len(audio)
    assert not np.array_equal(out[start:end], audio[start:end])


def test_snap_remove_boundaries_moves_to_low_amplitude_points() -> None:
    """Boundary snapping should move cuts to nearby low-amplitude samples."""
    audio = np.ones(1000, dtype=np.float32)
    audio[420] = 0.0
    audio[780] = 0.0
    snapped = _snap_remove_boundaries(audio, [(400, 800)], search_samples=30, match_samples=0)
    assert snapped == [(420, 780)]


def test_snap_remove_boundaries_alignment_refines_end() -> None:
    """Alignment should refine the end boundary to a better waveform match."""
    x = np.linspace(0, 8 * np.pi, 1600, endpoint=False)
    audio = np.sin(x).astype(np.float32)
    # Deliberately offset end away from phase-consistent continuation.
    snapped = _snap_remove_boundaries(audio, [(400, 710)], search_samples=40, match_samples=80)
    start, end = snapped[0]
    assert start >= 360
    assert end <= 750
    assert end != 710


# ---------------------------------------------------------------------------
# _normalized_cross_correlation
# ---------------------------------------------------------------------------


def test_ncc_perfect_match() -> None:
    """NCC should be 1.0 at the position of an exact match."""
    ref = _sine(1000, 0.5).astype(np.float64)
    offset_samples = int(0.5 * SAMPLE_RATE)
    audio = _concat(_silence(0.5), _sine(1000, 0.5), _silence(0.5)).astype(np.float64)
    ncc = _normalized_cross_correlation(audio, ref)
    assert ncc.max() == pytest.approx(1.0, abs=1e-5)
    # Peak should be at the start of the matching segment (~0.5 s offset)
    assert int(ncc.argmax()) == pytest.approx(offset_samples, abs=10)


def test_ncc_no_match() -> None:
    """NCC for a reference clip longer than the audio should return an empty array."""
    ref = _sine(440, 2.0).astype(np.float64)
    audio = _sine(440, 1.0).astype(np.float64)
    ncc = _normalized_cross_correlation(audio, ref)
    assert len(ncc) == 0


def test_ncc_silent_reference() -> None:
    """A zero-energy reference should return zeros, not raise."""
    ref = np.zeros(100, dtype=np.float64)
    audio = _sine(440, 1.0).astype(np.float64)
    ncc = _normalized_cross_correlation(audio, ref)
    assert np.all(ncc == 0.0)


# ---------------------------------------------------------------------------
# FingerprintDetector
# ---------------------------------------------------------------------------


def test_fingerprint_detector_finds_exact_clip() -> None:
    """Detector should locate an exact copy of the reference clip in the audio."""
    jingle = _sine(2000, 2.0)  # 2-second jingle at 2 kHz
    audio = _concat(_sine(440, 5.0), jingle, _sine(440, 5.0))
    det = FingerprintDetector(reference_clips=[jingle], correlation_threshold=0.9)
    segs = det.detect(audio, SAMPLE_RATE)
    assert len(segs) == 1
    start_s = segs[0][0] / SAMPLE_RATE
    end_s = segs[0][1] / SAMPLE_RATE
    assert start_s == pytest.approx(5.0, abs=0.1)
    assert end_s == pytest.approx(7.0, abs=0.1)


def test_fingerprint_detector_no_match() -> None:
    """Spectrally unrelated reference should not produce matches above threshold."""
    reference = _sine(2000, 1.0)  # 2 kHz reference
    audio = _sine(440, 10.0)  # 440 Hz audio only
    det = FingerprintDetector(reference_clips=[reference], correlation_threshold=0.9)
    segs = det.detect(audio, SAMPLE_RATE)
    assert segs == []


def test_fingerprint_detector_multiple_clips() -> None:
    """Detector should find occurrences of each reference clip independently."""
    jingle = _sine(2000, 1.0)
    sponsorship = _sine(3000, 1.0)
    audio = _concat(
        _sine(440, 3.0),
        jingle,
        _sine(440, 3.0),
        sponsorship,
        _sine(440, 3.0),
    )
    det = FingerprintDetector(reference_clips=[jingle, sponsorship], correlation_threshold=0.9)
    segs = det.detect(audio, SAMPLE_RATE)
    assert len(segs) == 2


def test_fingerprint_detector_stereo_audio() -> None:
    """Detector should handle stereo (2-D) audio correctly."""
    jingle = _sine(2000, 1.0)
    mono_audio = _concat(_sine(440, 3.0), jingle, _sine(440, 3.0))
    stereo = np.stack([mono_audio, mono_audio], axis=1)
    det = FingerprintDetector(reference_clips=[jingle], correlation_threshold=0.9)
    segs = det.detect(stereo, SAMPLE_RATE)
    assert len(segs) == 1


def test_fingerprint_detector_ref_longer_than_audio() -> None:
    """A reference clip longer than the audio should be silently skipped."""
    ref = _sine(440, 5.0)  # 5 s reference
    audio = _sine(440, 2.0)  # 2 s audio — shorter than reference
    det = FingerprintDetector(reference_clips=[ref], correlation_threshold=0.9)
    segs = det.detect(audio, SAMPLE_RATE)
    assert segs == []


# ---------------------------------------------------------------------------
# remove_ads — integration
# ---------------------------------------------------------------------------


def test_remove_ads_timestamps_strategy() -> None:
    """timestamps strategy should replace by default, preserving duration."""
    jingle = _sine(2000, 2.0)
    audio = _concat(_sine(440, 5.0), jingle, _sine(440, 5.0))
    out = remove_ads(
        audio,
        SAMPLE_RATE,
        strategy="timestamps",
        timestamps=[(5.0, 7.0)],
    )
    assert len(out) == len(audio)
    start = int(5.0 * SAMPLE_RATE)
    end = int(7.0 * SAMPLE_RATE)
    assert not np.array_equal(out[start:end], audio[start:end])


def test_remove_ads_timestamps_duck_action() -> None:
    """timestamps strategy supports explicit ducking."""
    jingle = _sine(2000, 2.0)
    audio = _concat(_sine(440, 5.0), jingle, _sine(440, 5.0))
    out = remove_ads(
        audio,
        SAMPLE_RATE,
        strategy="timestamps",
        timestamps=[(5.0, 7.0)],
        timestamp_action="duck",
        timestamp_duck_db=-18.0,
    )
    assert len(out) == len(audio)
    start = int(5.0 * SAMPLE_RATE)
    end = int(7.0 * SAMPLE_RATE)
    in_rms = np.sqrt(np.mean(audio[start:end] ** 2))
    out_rms = np.sqrt(np.mean(out[start:end] ** 2))
    assert out_rms < in_rms * 0.2


def test_remove_ads_timestamps_remove_action() -> None:
    """timestamps strategy can still cut intervals when explicitly requested."""
    jingle = _sine(2000, 2.0)
    audio = _concat(_sine(440, 5.0), jingle, _sine(440, 5.0))
    out = remove_ads(
        audio,
        SAMPLE_RATE,
        strategy="timestamps",
        timestamps=[(5.0, 7.0)],
        timestamp_action="remove",
    )
    assert len(out) < len(audio)
    assert abs(len(out) - 10 * SAMPLE_RATE) < SAMPLE_RATE * 0.1


def test_remove_ads_fingerprint_strategy() -> None:
    """remove_ads with fingerprint strategy should locate and remove the ad clip."""
    jingle = _sine(2000, 2.0)
    audio = _concat(_sine(440, 5.0), jingle, _sine(440, 5.0))
    out = remove_ads(
        audio,
        SAMPLE_RATE,
        strategy="fingerprint",
        reference_clips=[jingle],
        correlation_threshold=0.9,
    )
    assert len(out) < len(audio)


def test_remove_ads_combined_strategy() -> None:
    """combined strategy should apply both timestamps and fingerprint removal."""
    jingle = _sine(2000, 1.0)
    sponsorship = _sine(3000, 1.0)
    audio = _concat(
        _sine(440, 3.0),
        sponsorship,
        _sine(440, 3.0),
        jingle,
        _sine(440, 3.0),
    )
    out = remove_ads(
        audio,
        SAMPLE_RATE,
        strategy="combined",
        timestamps=[(3.0, 4.0)],  # remove the sponsorship read by timestamp
        reference_clips=[jingle],  # remove the jingle by fingerprint
        correlation_threshold=0.9,
    )
    assert len(out) < len(audio)


def test_fingerprint_detector_empty_reference_skipped() -> None:
    """An empty reference clip should be silently skipped."""
    empty_ref = np.array([], dtype=np.float32)
    audio = _sine(440, 3.0)
    det = FingerprintDetector(reference_clips=[empty_ref], correlation_threshold=0.9)
    segs = det.detect(audio, SAMPLE_RATE)
    assert segs == []
    """When nothing is detected, the output should be identical to the input."""
    audio = _sine(440, 3.0, amp=0.3)
    out = remove_ads(audio, SAMPLE_RATE, strategy="timestamps")
    np.testing.assert_array_equal(out, audio)


def test_remove_ads_stereo_timestamps() -> None:
    """remove_ads should handle stereo replacement with timestamps by default."""
    mono_audio = _concat(_sine(440, 3.0), _sine(2000, 2.0), _sine(440, 3.0))
    stereo = np.stack([mono_audio, mono_audio], axis=1)
    out = remove_ads(stereo, SAMPLE_RATE, strategy="timestamps", timestamps=[(3.0, 5.0)])
    assert out.ndim == 2
    assert out.shape[0] == stereo.shape[0]


# ---------------------------------------------------------------------------
# remove_ads — validation errors
# ---------------------------------------------------------------------------


def test_remove_ads_invalid_sample_rate() -> None:
    audio = _sine(440, 1.0)
    with pytest.raises(ValueError, match="sample_rate"):
        remove_ads(audio, 0)


def test_remove_ads_empty_audio() -> None:
    with pytest.raises(ValueError, match="empty"):
        remove_ads(np.array([], dtype=np.float32), SAMPLE_RATE)


def test_remove_ads_invalid_fade_ms() -> None:
    audio = _sine(440, 1.0)
    with pytest.raises(ValueError, match="fade_ms"):
        remove_ads(audio, SAMPLE_RATE, fade_ms=-1.0)


def test_remove_ads_invalid_correlation_threshold() -> None:
    audio = _sine(440, 1.0)
    with pytest.raises(ValueError, match="correlation_threshold"):
        remove_ads(audio, SAMPLE_RATE, correlation_threshold=-0.1)
    with pytest.raises(ValueError, match="correlation_threshold"):
        remove_ads(audio, SAMPLE_RATE, correlation_threshold=1.1)


def test_remove_ads_timestamp_end_before_start() -> None:
    audio = _sine(440, 5.0)
    with pytest.raises(ValueError, match="greater than start"):
        remove_ads(audio, SAMPLE_RATE, strategy="timestamps", timestamps=[(3.0, 1.0)])


def test_remove_ads_negative_timestamp_start() -> None:
    """A negative timestamp start should raise a ValueError."""
    audio = _sine(440, 5.0)
    with pytest.raises(ValueError, match="timestamp start must be >= 0"):
        remove_ads(audio, SAMPLE_RATE, strategy="timestamps", timestamps=[(-1.0, 2.0)])


def test_remove_ads_timestamps_beyond_audio_duration() -> None:
    """Out-of-range timestamps should be clamped and replaced in-place."""
    audio = _sine(440, 3.0)  # 3 s audio
    # end_s (10.0) is beyond the audio length; only content up to the end is replaced
    out = remove_ads(
        audio,
        SAMPLE_RATE,
        strategy="timestamps",
        timestamps=[(2.0, 10.0)],
    )
    assert len(out) == len(audio)
    start = int(2.0 * SAMPLE_RATE)
    assert not np.array_equal(out[start:], audio[start:])


def test_remove_ads_duck_action_rejects_positive_db() -> None:
    """Positive duck dB should be rejected to avoid accidental ad boosting."""
    audio = _sine(440, 3.0)
    with pytest.raises(ValueError, match="timestamp_duck_db"):
        remove_ads(
            audio,
            SAMPLE_RATE,
            strategy="timestamps",
            timestamps=[(1.0, 2.0)],
            timestamp_action="duck",
            timestamp_duck_db=6.0,
        )


def test_remove_ads_invalid_cut_snap_ms() -> None:
    """Negative cut_snap_ms should be rejected."""
    audio = _sine(440, 3.0)
    with pytest.raises(ValueError, match="cut_snap_ms"):
        remove_ads(
            audio,
            SAMPLE_RATE,
            strategy="timestamps",
            timestamps=[(1.0, 2.0)],
            timestamp_action="remove",
            cut_snap_ms=-10.0,
        )


def test_remove_ads_invalid_cut_match_ms() -> None:
    """Negative cut_match_ms should be rejected."""
    audio = _sine(440, 3.0)
    with pytest.raises(ValueError, match="cut_match_ms"):
        remove_ads(
            audio,
            SAMPLE_RATE,
            strategy="timestamps",
            timestamps=[(1.0, 2.0)],
            timestamp_action="remove",
            cut_match_ms=-5.0,
        )
