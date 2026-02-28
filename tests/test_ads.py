"""Unit tests for audio_cleaner.ads — ad/interrupt detection and removal."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from audio_cleaner.ads import (
    FingerprintDetector,
    _apply_crossfade,
    _apply_ducking,
    _apply_replacement,
    _compute_spectral_signature,
    _merge_segments,
    _normalized_cross_correlation,
    _resample_audio,
    _rms_normalize,
    _snap_remove_boundaries,
    _spectral_similarity,
    clean_with_profile,
    create_ad_profile,
    load_ad_profile,
    remove_ads,
    save_ad_profile,
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


# ---------------------------------------------------------------------------
# _merge_segments — gap_samples parameter
# ---------------------------------------------------------------------------


def test_merge_segments_with_gap_merges_nearby() -> None:
    """Segments within gap_samples of each other should be merged."""
    segs = [(0, 100), (150, 250)]
    # gap = 50 samples, gap_samples = 60 → should merge
    merged = _merge_segments(segs, gap_samples=60)
    assert merged == [(0, 250)]


def test_merge_segments_with_gap_keeps_far_apart() -> None:
    """Segments farther apart than gap_samples should remain separate."""
    segs = [(0, 100), (300, 400)]
    merged = _merge_segments(segs, gap_samples=50)
    assert merged == segs


def test_merge_segments_zero_gap_unchanged() -> None:
    """gap_samples=0 should behave identically to the original implementation."""
    segs = [(0, 100), (101, 200)]
    # gap = 1 sample; with gap_samples=0 they should NOT merge
    assert _merge_segments(segs, gap_samples=0) == segs


# ---------------------------------------------------------------------------
# _rms_normalize
# ---------------------------------------------------------------------------


def test_rms_normalize_unit_rms() -> None:
    """Normalized audio should have RMS ≈ 1."""
    audio = _sine(440, 1.0, amp=0.3)
    out = _rms_normalize(audio)
    rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    assert rms == pytest.approx(1.0, abs=1e-5)


def test_rms_normalize_silent_unchanged() -> None:
    """A silent clip should be returned as-is (no divide-by-zero)."""
    audio = np.zeros(1000, dtype=np.float32)
    out = _rms_normalize(audio)
    np.testing.assert_array_equal(out, audio)


# ---------------------------------------------------------------------------
# _compute_spectral_signature
# ---------------------------------------------------------------------------


def test_spectral_signature_shape() -> None:
    """Spectral signature should have the requested number of bins."""
    audio = _sine(1000, 1.0)
    sig = _compute_spectral_signature(audio, SAMPLE_RATE, n_bins=32)
    assert sig.shape == (32,)
    assert sig.dtype == np.float32


def test_spectral_signature_short_clip_returns_zeros() -> None:
    """A clip shorter than the FFT window should return zeros."""
    audio = np.zeros(4, dtype=np.float32)
    sig = _compute_spectral_signature(audio, SAMPLE_RATE, n_bins=32)
    assert sig.shape == (32,)
    assert np.all(sig == 0.0)


def test_spectral_signature_different_tones_differ() -> None:
    """Spectral signatures for different tones should not be identical."""
    sig_low = _compute_spectral_signature(_sine(200, 1.0), SAMPLE_RATE, n_bins=32)
    sig_high = _compute_spectral_signature(_sine(4000, 1.0), SAMPLE_RATE, n_bins=32)
    assert not np.allclose(sig_low, sig_high)


# ---------------------------------------------------------------------------
# _spectral_similarity
# ---------------------------------------------------------------------------


def test_spectral_similarity_same_clip_high() -> None:
    """Spectral similarity of a clip with itself should be ~1.0."""
    audio = _sine(1000, 1.0)
    sig = _compute_spectral_signature(audio, SAMPLE_RATE)
    score = _spectral_similarity(audio, sig, SAMPLE_RATE)
    assert score == pytest.approx(1.0, abs=0.05)


def test_spectral_similarity_different_clips_lower() -> None:
    """Spectral similarity between very different tones should be less than 1.0."""
    clip_a = _sine(440, 1.0)
    clip_b = _sine(4000, 1.0)
    sig_a = _compute_spectral_signature(clip_a, SAMPLE_RATE)
    score = _spectral_similarity(clip_b, sig_a, SAMPLE_RATE)
    assert score < 0.9


# ---------------------------------------------------------------------------
# _resample_audio
# ---------------------------------------------------------------------------


def test_resample_audio_same_rate_returns_original() -> None:
    """Resampling to the same rate should return the original array."""
    audio = _sine(440, 1.0)
    out = _resample_audio(audio, SAMPLE_RATE, SAMPLE_RATE)
    np.testing.assert_array_equal(out, audio)


def test_resample_audio_changes_length() -> None:
    """Resampling to a different rate should change the array length."""
    audio = _sine(440, 1.0, sr=SAMPLE_RATE)
    target_sr = 8000
    out = _resample_audio(audio, SAMPLE_RATE, target_sr)
    expected_len = int(len(audio) * target_sr / SAMPLE_RATE)
    assert abs(len(out) - expected_len) <= 2  # allow rounding differences


# ---------------------------------------------------------------------------
# create_ad_profile
# ---------------------------------------------------------------------------


def _build_audio_with_repeated_ad(
    ad_clip: np.ndarray,
    ad_start_times_s: list[float],
    total_s: float,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Build a synthetic recording with *ad_clip* inserted at given start times."""
    audio = _sine(440, total_s, sr=sr)
    for t in ad_start_times_s:
        s = int(t * sr)
        e = min(len(audio), s + len(ad_clip))
        audio[s:e] = ad_clip[: e - s]
    return audio


def test_create_ad_profile_basic() -> None:
    """create_ad_profile should produce one fingerprint per valid interval."""
    ad = _sine(2000, 2.0)  # 2-second ad at 2 kHz
    audio = _build_audio_with_repeated_ad(ad, [10.0, 30.0, 50.0, 70.0], total_s=90.0)
    profile = create_ad_profile(
        audio,
        SAMPLE_RATE,
        rough_timestamps=[(10.0, 12.0), (30.0, 32.0), (50.0, 52.0), (70.0, 72.0)],
        created_from="test.flac",
    )
    assert len(profile.fingerprints) == 4
    assert profile.sample_rate == SAMPLE_RATE
    assert profile.created_from == "test.flac"
    assert profile.profile_version == 1


def test_create_ad_profile_duration_stored() -> None:
    """Fingerprint duration should match the extracted clip length."""
    ad = _sine(2000, 2.0)
    audio = _build_audio_with_repeated_ad(ad, [5.0], total_s=20.0)
    profile = create_ad_profile(audio, SAMPLE_RATE, rough_timestamps=[(5.0, 7.0)])
    fp = profile.fingerprints[0]
    assert fp.duration_s == pytest.approx(2.0, abs=0.1)


def test_create_ad_profile_skips_short_clips() -> None:
    """Intervals shorter than 0.5 s should produce no fingerprint."""
    audio = _sine(440, 10.0)
    profile = create_ad_profile(
        audio,
        SAMPLE_RATE,
        rough_timestamps=[(2.0, 2.3)],  # only 0.3 s — below minimum
    )
    assert len(profile.fingerprints) == 0


def test_create_ad_profile_threshold_calibrated() -> None:
    """NCC threshold should be calibrated from pairwise similarities (≥ floor)."""
    ad = _sine(2000, 2.0)
    audio = _build_audio_with_repeated_ad(ad, [5.0, 20.0, 35.0, 50.0], total_s=60.0)
    profile = create_ad_profile(
        audio,
        SAMPLE_RATE,
        rough_timestamps=[(5.0, 7.0), (20.0, 22.0), (35.0, 37.0), (50.0, 52.0)],
    )
    for fp in profile.fingerprints:
        assert fp.ncc_threshold >= 0.65  # floor


def test_create_ad_profile_stereo_input() -> None:
    """create_ad_profile should accept 2-D (stereo) audio."""
    mono = _sine(440, 10.0)
    ad_mono = _sine(2000, 1.0)
    s = int(3.0 * SAMPLE_RATE)
    mono[s : s + len(ad_mono)] = ad_mono
    stereo = np.stack([mono, mono], axis=1)
    profile = create_ad_profile(stereo, SAMPLE_RATE, rough_timestamps=[(3.0, 4.0)])
    assert len(profile.fingerprints) == 1


def test_create_ad_profile_invalid_sample_rate() -> None:
    """Negative sample rate should raise ValueError."""
    audio = _sine(440, 1.0)
    with pytest.raises(ValueError, match="sample_rate"):
        create_ad_profile(audio, -1, rough_timestamps=[(0.0, 0.5)])


def test_create_ad_profile_empty_audio() -> None:
    """Empty audio should raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        create_ad_profile(np.array([], dtype=np.float32), SAMPLE_RATE, rough_timestamps=[])


# ---------------------------------------------------------------------------
# save_ad_profile / load_ad_profile
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path: Path) -> None:
    """Profile saved with save_ad_profile must be loadable and identical."""
    ad = _sine(2000, 2.0)
    audio = _build_audio_with_repeated_ad(ad, [5.0, 20.0], total_s=30.0)
    profile = create_ad_profile(
        audio,
        SAMPLE_RATE,
        rough_timestamps=[(5.0, 7.0), (20.0, 22.0)],
        created_from="source.flac",
    )

    base = tmp_path / "my_profile"
    save_ad_profile(profile, base)

    assert (tmp_path / "my_profile.json").exists()
    assert (tmp_path / "my_profile.npz").exists()

    loaded = load_ad_profile(base)
    assert loaded.profile_version == profile.profile_version
    assert loaded.sample_rate == profile.sample_rate
    assert loaded.created_from == profile.created_from
    assert len(loaded.fingerprints) == len(profile.fingerprints)
    for orig_fp, loaded_fp in zip(profile.fingerprints, loaded.fingerprints, strict=True):
        assert orig_fp.id == loaded_fp.id
        assert orig_fp.duration_s == pytest.approx(loaded_fp.duration_s, abs=1e-6)
        assert orig_fp.ncc_threshold == pytest.approx(loaded_fp.ncc_threshold, abs=1e-6)
        np.testing.assert_allclose(orig_fp.template, loaded_fp.template, atol=1e-6)
        np.testing.assert_allclose(
            orig_fp.spectral_signature, loaded_fp.spectral_signature, atol=1e-6
        )


def test_save_load_path_with_extension(tmp_path: Path) -> None:
    """save/load should work even when path is passed with .json extension."""
    ad = _sine(2000, 1.5)
    audio = _build_audio_with_repeated_ad(ad, [3.0], total_s=10.0)
    profile = create_ad_profile(audio, SAMPLE_RATE, rough_timestamps=[(3.0, 4.5)])

    save_ad_profile(profile, tmp_path / "profile.json")
    loaded = load_ad_profile(tmp_path / "profile.json")
    assert len(loaded.fingerprints) == 1


def test_load_missing_file_raises(tmp_path: Path) -> None:
    """Loading a non-existent profile should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_ad_profile(tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# clean_with_profile
# ---------------------------------------------------------------------------


def test_clean_with_profile_remove_detects_ad() -> None:
    """clean_with_profile with action='remove' should shorten the audio."""
    ad = _sine(2000, 2.0)
    audio = _concat(_sine(440, 5.0), ad, _sine(440, 5.0))
    profile = create_ad_profile(audio, SAMPLE_RATE, rough_timestamps=[(5.0, 7.0)])
    assert len(profile.fingerprints) > 0

    # Target: same structure — ad at 5-7 s
    target = _concat(_sine(440, 5.0), ad, _sine(440, 5.0))
    cleaned = clean_with_profile(target, SAMPLE_RATE, profile, action="remove")
    assert len(cleaned) < len(target)


def test_clean_with_profile_replace_preserves_length() -> None:
    """clean_with_profile with action='replace' should preserve audio length."""
    ad = _sine(2000, 2.0)
    audio = _concat(_sine(440, 5.0), ad, _sine(440, 5.0))
    profile = create_ad_profile(audio, SAMPLE_RATE, rough_timestamps=[(5.0, 7.0)])

    cleaned = clean_with_profile(audio, SAMPLE_RATE, profile, action="replace")
    assert len(cleaned) == len(audio)
    # Content in the ad region should have changed
    s, e = int(5.0 * SAMPLE_RATE), int(7.0 * SAMPLE_RATE)
    assert not np.array_equal(cleaned[s:e], audio[s:e])


def test_clean_with_profile_duck_preserves_length() -> None:
    """clean_with_profile with action='duck' should preserve audio length."""
    ad = _sine(2000, 2.0)
    audio = _concat(_sine(440, 5.0), ad, _sine(440, 5.0))
    profile = create_ad_profile(audio, SAMPLE_RATE, rough_timestamps=[(5.0, 7.0)])

    cleaned = clean_with_profile(audio, SAMPLE_RATE, profile, action="duck")
    assert len(cleaned) == len(audio)
    s, e = int(5.0 * SAMPLE_RATE), int(7.0 * SAMPLE_RATE)
    in_rms = float(np.sqrt(np.mean(audio[s:e].astype(np.float64) ** 2)))
    out_rms = float(np.sqrt(np.mean(cleaned[s:e].astype(np.float64) ** 2)))
    assert out_rms < in_rms * 0.5


def test_clean_with_profile_no_match_returns_copy() -> None:
    """When no ad is found, clean_with_profile should return a copy of the input."""
    ad = _sine(2000, 2.0)
    ref = _concat(_sine(440, 5.0), ad, _sine(440, 5.0))
    profile = create_ad_profile(ref, SAMPLE_RATE, rough_timestamps=[(5.0, 7.0)])

    # Completely different audio — no ad
    target = _sine(880, 10.0)
    import warnings

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        cleaned = clean_with_profile(target, SAMPLE_RATE, profile)
    np.testing.assert_array_equal(cleaned, target)


def test_clean_with_profile_four_ad_intervals() -> None:
    """Sparse-data (4 examples) profile should detect all four ad insertions."""
    ad = _sine(2000, 2.0)
    ad_times = [10.0, 30.0, 55.0, 80.0]
    audio = _build_audio_with_repeated_ad(ad, ad_times, total_s=90.0)

    profile = create_ad_profile(
        audio,
        SAMPLE_RATE,
        rough_timestamps=[(t, t + 2.0) for t in ad_times],
    )
    assert len(profile.fingerprints) == 4

    target = _build_audio_with_repeated_ad(ad, ad_times, total_s=90.0)
    cleaned = clean_with_profile(target, SAMPLE_RATE, profile, action="remove")
    # Should be noticeably shorter (>= 4 x ~2 s removed)
    assert len(audio) - len(cleaned) >= SAMPLE_RATE * 4


def test_clean_with_profile_drift_tolerance() -> None:
    """Detection should be robust to ±300 ms timing drift in target file."""
    ad = _sine(2000, 2.0)
    # Learn from exact positions
    ref_audio = _build_audio_with_repeated_ad(ad, [10.0, 25.0], total_s=40.0)
    profile = create_ad_profile(
        ref_audio,
        SAMPLE_RATE,
        rough_timestamps=[(10.0, 12.0), (25.0, 27.0)],
    )

    # Apply with 300 ms drift
    drifted_audio = _build_audio_with_repeated_ad(ad, [10.3, 25.3], total_s=40.0)
    cleaned = clean_with_profile(drifted_audio, SAMPLE_RATE, profile, action="remove")
    assert len(cleaned) < len(drifted_audio)


def test_clean_with_profile_cross_sample_rate() -> None:
    """Profile learned at one sample rate should work on audio at another rate."""
    sr_learn = 16000
    sr_apply = 8000

    ad = _sine(2000, 1.0, sr=sr_learn)
    ref_audio = _concat(
        _sine(440, 3.0, sr=sr_learn), ad, _sine(440, 3.0, sr=sr_learn)
    )
    profile = create_ad_profile(ref_audio, sr_learn, rough_timestamps=[(3.0, 4.0)])

    # Target at half the sample rate
    ad_low = _sine(2000, 1.0, sr=sr_apply)
    target_audio = _concat(
        _sine(440, 3.0, sr=sr_apply), ad_low, _sine(440, 3.0, sr=sr_apply)
    )
    cleaned = clean_with_profile(target_audio, sr_apply, profile, action="remove")
    assert len(cleaned) < len(target_audio)


def test_clean_with_profile_invalid_action() -> None:
    """Unknown action should raise ValueError."""
    audio = _sine(440, 3.0)
    ad = _sine(2000, 1.0)
    src = _concat(_sine(440, 1.0), ad, _sine(440, 1.0))
    profile = create_ad_profile(src, SAMPLE_RATE, rough_timestamps=[(1.0, 2.0)])
    with pytest.raises(ValueError, match="action"):
        clean_with_profile(audio, SAMPLE_RATE, profile, action="silence")  # type: ignore[arg-type]


def test_clean_with_profile_invalid_duck_db() -> None:
    """Positive duck_db should raise ValueError."""
    audio = _sine(440, 3.0)
    ad = _sine(2000, 1.0)
    src = _concat(_sine(440, 1.0), ad, _sine(440, 1.0))
    profile = create_ad_profile(src, SAMPLE_RATE, rough_timestamps=[(1.0, 2.0)])
    with pytest.raises(ValueError, match="duck_db"):
        clean_with_profile(audio, SAMPLE_RATE, profile, action="duck", duck_db=6.0)


def test_clean_with_profile_stereo_audio() -> None:
    """clean_with_profile should accept stereo (2-D) audio."""
    ad = _sine(2000, 1.5)
    mono = _concat(_sine(440, 3.0), ad, _sine(440, 3.0))
    profile = create_ad_profile(mono, SAMPLE_RATE, rough_timestamps=[(3.0, 4.5)])
    stereo = np.stack([mono, mono], axis=1)
    cleaned = clean_with_profile(stereo, SAMPLE_RATE, profile, action="replace")
    assert cleaned.ndim == 2
    assert cleaned.shape[0] == stereo.shape[0]


def test_clean_with_profile_empty_audio() -> None:
    """Empty audio should raise ValueError."""
    ad = _sine(2000, 1.0)
    src = _concat(_sine(440, 1.0), ad, _sine(440, 1.0))
    profile = create_ad_profile(src, SAMPLE_RATE, rough_timestamps=[(1.0, 2.0)])
    with pytest.raises(ValueError, match="empty"):
        clean_with_profile(np.array([], dtype=np.float32), SAMPLE_RATE, profile)


# ---------------------------------------------------------------------------
# End-to-end CLI: learn-ads + apply-ads-profile
# ---------------------------------------------------------------------------


def test_cli_learn_and_apply(tmp_path: Path) -> None:
    """End-to-end: learn-ads + apply-ads-profile via Python entry-points."""
    import soundfile as sf

    from audio_cleaner.__main__ import main

    ad = _sine(2000, 2.0)
    audio = _build_audio_with_repeated_ad(ad, [5.0, 20.0], total_s=30.0)
    in_file = tmp_path / "source.wav"
    sf.write(str(in_file), audio, SAMPLE_RATE)

    profile_base = str(tmp_path / "profile")
    out_dir = tmp_path / "out"

    # Step 1: learn-ads
    import sys

    sys_argv_backup = sys.argv
    try:
        sys.argv = [
            "audio-cleaner",
            "learn-ads",
            "--input", str(in_file),
            "--timestamps", "5.0,7.0", "20.0,22.0",
            "--profile-out", profile_base,
        ]
        try:
            main()
        except SystemExit as exc:
            assert exc.code == 0, f"learn-ads exited with code {exc.code}"
    finally:
        sys.argv = sys_argv_backup

    assert (tmp_path / "profile.json").exists()
    assert (tmp_path / "profile.npz").exists()

    # Step 2: apply-ads-profile
    try:
        sys.argv = [
            "audio-cleaner",
            "apply-ads-profile",
            "--input", str(in_file),
            "--profile", profile_base,
            "--output", str(out_dir),
            "--action", "remove",
        ]
        try:
            main()
        except SystemExit as exc:
            assert exc.code == 0, f"apply-ads-profile exited with code {exc.code}"
    finally:
        sys.argv = sys_argv_backup

    out_file = out_dir / "source.wav"
    assert out_file.exists()
    cleaned, _sr = sf.read(str(out_file), dtype="float32")
    assert len(cleaned) < len(audio)
