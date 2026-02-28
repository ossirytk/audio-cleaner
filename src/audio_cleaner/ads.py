"""Ad / interrupt detection and removal for FLAC and WAV audio files.

Supports removal of advertisements, station IDs / jingles, and sponsorship reads
from audio recordings using either user-specified timestamps or audio fingerprinting
against known reference clips.

Methods
-------
- Timestamp-based removal: user provides (start_s, end_s) intervals to remove.
- Fingerprint-based detection: normalized cross-correlation against reference clips.
- Clean segment reassembly with raised-cosine crossfade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

AudioArray = npt.NDArray[np.float32]
Segment = tuple[int, int]  # (start_sample, end_sample) inclusive start, exclusive end


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _raised_cosine_fade(length: int) -> AudioArray:
    """Return a raised-cosine (Hann) fade-in window of *length* samples.

    Args:
        length: Number of samples in the fade window.

    Returns:
        1-D float32 array with values in [0, 1].
    """
    t = np.linspace(0.0, np.pi, length)
    return ((1.0 - np.cos(t)) / 2.0).astype(np.float32)


def _apply_crossfade(
    audio: AudioArray,
    remove_segments: list[Segment],
    sample_rate: int,
    fade_ms: float = 20.0,
) -> AudioArray:
    """Remove *remove_segments* from *audio* and apply crossfades at cut points.

    Args:
        audio: 1-D float32 audio array (mono) or 2-D (samples x channels).
        remove_segments: Sorted, non-overlapping list of (start, end) sample pairs to remove.
        sample_rate: Sample rate in Hz.
        fade_ms: Duration of the raised-cosine fade at each cut point in milliseconds.

    Returns:
        Reconstructed float32 array with the specified segments removed.
    """
    if not remove_segments:
        return audio.copy()

    fade_samples = max(1, int(sample_rate * fade_ms / 1000.0))
    total_samples = audio.shape[0]
    mono = audio.ndim == 1

    # Build list of kept (start, end) regions
    kept: list[Segment] = []
    cursor = 0
    for seg_start, seg_end in sorted(remove_segments):
        if cursor < seg_start:
            kept.append((cursor, seg_start))
        cursor = seg_end
    if cursor < total_samples:
        kept.append((cursor, total_samples))

    if not kept:
        return np.zeros((0,) if mono else (0, audio.shape[1]), dtype=np.float32)

    out: AudioArray = audio[kept[0][0] : kept[0][1]].copy()

    for start, end in kept[1:]:
        next_chunk: AudioArray = audio[start:end].copy()
        overlap_len = min(fade_samples, out.shape[0], next_chunk.shape[0])

        if overlap_len > 0:
            fade_in = _raised_cosine_fade(overlap_len)
            fade_out = fade_in[::-1].copy()

            if mono:
                overlap = out[-overlap_len:] * fade_out + next_chunk[:overlap_len] * fade_in
            else:
                overlap = (
                    out[-overlap_len:] * fade_out[:, np.newaxis]
                    + next_chunk[:overlap_len] * fade_in[:, np.newaxis]
                )
            out = np.concatenate([out[:-overlap_len], overlap, next_chunk[overlap_len:]], axis=0)
        else:
            out = np.concatenate([out, next_chunk], axis=0)

    return out


def _apply_ducking(
    audio: AudioArray,
    duck_segments: list[Segment],
    sample_rate: int,
    duck_db: float,
    fade_ms: float = 20.0,
) -> AudioArray:
    """Attenuate *duck_segments* in *audio* with smooth fade boundaries.

    Args:
        audio: 1-D float32 audio array (mono) or 2-D (samples x channels).
        duck_segments: Sorted, non-overlapping list of (start, end) sample pairs to attenuate.
        sample_rate: Sample rate in Hz.
        duck_db: Gain applied to each segment in dB (negative attenuates, positive boosts).
        fade_ms: Duration of fade ramps at each segment boundary in milliseconds.

    Returns:
        Float32 audio array with the selected segments attenuated while preserving length.
    """
    if not duck_segments:
        return audio.copy()

    out = audio.copy()
    mono = out.ndim == 1
    total_samples = out.shape[0]
    fade_samples = max(1, int(sample_rate * fade_ms / 1000.0))
    attenuation = float(10.0 ** (duck_db / 20.0))

    for start, end in duck_segments:
        clamped_start = max(0, start)
        clamped_end = min(total_samples, end)
        if clamped_start >= clamped_end:
            continue

        seg_len = clamped_end - clamped_start
        ramp_len = min(fade_samples, seg_len // 2)

        if ramp_len > 0:
            ramp = _raised_cosine_fade(ramp_len)
            fade_in = 1.0 - (1.0 - attenuation) * ramp
            fade_out = fade_in[::-1]
            mid_start = clamped_start + ramp_len
            mid_end = clamped_end - ramp_len

            if mono:
                out[clamped_start:mid_start] *= fade_in
                if mid_start < mid_end:
                    out[mid_start:mid_end] *= attenuation
                out[mid_end:clamped_end] *= fade_out
            else:
                out[clamped_start:mid_start] *= fade_in[:, np.newaxis]
                if mid_start < mid_end:
                    out[mid_start:mid_end] *= attenuation
                out[mid_end:clamped_end] *= fade_out[:, np.newaxis]
        else:
            if mono:
                out[clamped_start:clamped_end] *= attenuation
            else:
                out[clamped_start:clamped_end] *= attenuation

    return out


def _apply_replacement(
    audio: AudioArray,
    replace_segments: list[Segment],
) -> AudioArray:
    """Replace *replace_segments* with smooth bridges while preserving duration.

    Each marked segment is replaced by a raised-cosine interpolation between the
    sample immediately before the segment and the sample immediately after it.

    Args:
        audio: 1-D float32 audio array (mono) or 2-D (samples x channels).
        replace_segments: Sorted, non-overlapping list of (start, end) sample pairs.

    Returns:
        Float32 audio array with replaced segments and unchanged total length.
    """
    if not replace_segments:
        return audio.copy()

    out = audio.copy()
    mono = out.ndim == 1
    total_samples = out.shape[0]

    for start, end in replace_segments:
        clamped_start = max(0, start)
        clamped_end = min(total_samples, end)
        seg_len = clamped_end - clamped_start
        if seg_len <= 0:
            continue

        if clamped_start > 0:
            left = out[clamped_start - 1]
        elif clamped_end < total_samples:
            left = out[clamped_end]
        else:
            left = 0.0 if mono else np.zeros(out.shape[1], dtype=np.float32)

        if clamped_end < total_samples:
            right = out[clamped_end]
        else:
            right = left

        ramp = _raised_cosine_fade(seg_len)
        if mono:
            out[clamped_start:clamped_end] = left * (1.0 - ramp) + right * ramp
        else:
            out[clamped_start:clamped_end] = (
                left * (1.0 - ramp[:, np.newaxis]) + right * ramp[:, np.newaxis]
            )

    return out


def _snap_remove_boundaries(
    audio: AudioArray,
    remove_segments: list[Segment],
    search_samples: int,
    match_samples: int,
) -> list[Segment]:
    """Snap remove boundaries to nearby low-amplitude samples.

    Args:
        audio: 1-D float32 audio array (mono) or 2-D (samples x channels).
        remove_segments: Sorted, non-overlapping list of (start, end) sample pairs.
        search_samples: Half-window size in samples for boundary search.
        match_samples: Window length used to align post-cut audio with pre-cut context.

    Returns:
        Updated segment list with boundaries moved to nearby low-amplitude points.
    """
    if not remove_segments:
        return remove_segments

    mono = audio if audio.ndim == 1 else audio.mean(axis=1).astype(np.float32)
    total = mono.shape[0]

    def _nearest_low(idx: int) -> int:
        if search_samples <= 0:
            return max(0, min(idx, total - 1))
        lo = max(0, idx - search_samples)
        hi = min(total, idx + search_samples + 1)
        if lo >= hi:
            return idx
        local = mono[lo:hi]
        return lo + int(np.argmin(np.abs(local)))

    def _align_end(start_idx: int, end_idx: int) -> int:
        if search_samples <= 0 or match_samples <= 0:
            return end_idx

        pre_hi = start_idx
        pre_lo = max(0, pre_hi - match_samples)
        pre = mono[pre_lo:pre_hi]
        window_len = pre.shape[0]
        if window_len < 8:
            return end_idx

        candidate_lo = max(start_idx + 1, end_idx - search_samples)
        candidate_hi = min(total - window_len, end_idx + search_samples)
        if candidate_hi < candidate_lo:
            return end_idx

        region = mono[candidate_lo : candidate_hi + window_len]
        if region.shape[0] < window_len:
            return end_idx

        pre_energy = float(np.dot(pre, pre))
        region_sq = region**2
        cumsum = np.concatenate([[0.0], np.cumsum(region_sq, dtype=np.float64)])
        n_positions = candidate_hi - candidate_lo + 1
        region_energy = cumsum[window_len:] - cumsum[:n_positions]
        corr = np.correlate(region, pre, mode="valid")
        mse = pre_energy + region_energy - (2.0 * corr)
        best_offset = int(np.argmin(mse))
        return candidate_lo + best_offset

    snapped: list[Segment] = []
    for start, end in remove_segments:
        snapped_start = _nearest_low(start)
        snapped_end = _nearest_low(end)
        snapped_end = _align_end(snapped_start, snapped_end)
        if snapped_end <= snapped_start:
            snapped_start = max(0, min(start, total - 1))
            snapped_end = max(snapped_start + 1, min(end, total))
        snapped.append((snapped_start, snapped_end))

    return _merge_segments(snapped)


# ---------------------------------------------------------------------------
# Fingerprint detector
# ---------------------------------------------------------------------------


def _normalized_cross_correlation(
    audio: npt.NDArray[np.float64],
    reference: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute normalized cross-correlation between *audio* and *reference*.

    Args:
        audio: 1-D float64 audio array.
        reference: 1-D float64 reference clip, must be shorter than *audio*.

    Returns:
        Array of NCC values at each valid position
        (length = ``len(audio) - len(reference) + 1``).  Values are in [-1, 1].
    """
    from scipy.signal import correlate  # type: ignore[attr-defined]

    ref_len = len(reference)
    n_positions = len(audio) - ref_len + 1
    if n_positions <= 0:
        return np.array([], dtype=np.float64)

    ref_energy = float(np.sqrt(np.dot(reference, reference)))
    if ref_energy < 1e-10:
        return np.zeros(n_positions, dtype=np.float64)

    corr = correlate(audio, reference, mode="valid")

    # Compute rolling L2 norm of audio windows using prefix sums
    audio_sq = audio**2
    cumsum = np.concatenate([[0.0], np.cumsum(audio_sq)])
    window_energies = np.sqrt(np.maximum(0.0, cumsum[ref_len:] - cumsum[:n_positions]))

    denom = ref_energy * window_energies
    with np.errstate(divide="ignore", invalid="ignore"):
        ncc = np.where(denom > 1e-10, corr / denom, 0.0)
    return ncc


@dataclass
class FingerprintDetector:
    """Detect ad/jingle segments by matching against known reference audio clips.

    Uses normalized cross-correlation to locate occurrences of known
    advertisement, station ID / jingle, or sponsorship read clips within the audio.

    Args:
        reference_clips: List of known ad/jingle audio arrays (1-D or 2-D float32).
        correlation_threshold: Minimum normalized cross-correlation score to flag
            a match (default: 0.7).
    """

    reference_clips: list[AudioArray]
    correlation_threshold: float = 0.7

    def detect(self, audio: AudioArray, sample_rate: int) -> list[Segment]:
        """Return a list of (start, end) sample pairs that match a reference clip.

        Args:
            audio: 1-D or 2-D float32 audio array.
            sample_rate: Sample rate in Hz.  Retained for interface consistency;
                the correlation calculation does not depend on it.

        Returns:
            Sorted list of (start_sample, end_sample) matched segments.
        """
        mono = audio if audio.ndim == 1 else audio.mean(axis=1).astype(np.float32)
        audio_f64 = mono.astype(np.float64)

        segments: list[Segment] = []
        for ref in self.reference_clips:
            ref_mono = ref if ref.ndim == 1 else ref.mean(axis=1).astype(np.float32)
            ref_f64 = ref_mono.astype(np.float64)
            ref_len = len(ref_f64)
            if ref_len == 0 or ref_len > len(audio_f64):
                continue

            ncc = _normalized_cross_correlation(audio_f64, ref_f64)
            n = len(ncc)
            # Collect non-overlapping matches: find all candidate positions above
            # threshold in a single vectorised pass, then greedily select them.
            candidate_indices = np.where(ncc >= self.correlation_threshold)[0]
            next_allowed_pos = 0
            for idx in candidate_indices:
                if idx < next_allowed_pos:
                    continue  # overlaps with previous match
                end_search = min(int(idx) + ref_len, n)
                best_pos = int(idx) + int(np.argmax(ncc[idx:end_search]))
                segments.append((best_pos, best_pos + ref_len))
                next_allowed_pos = best_pos + ref_len

        return sorted(segments)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _merge_segments(segments: list[Segment]) -> list[Segment]:
    """Merge overlapping or adjacent segments into minimal non-overlapping intervals.

    Args:
        segments: List of (start, end) sample pairs.

    Returns:
        Sorted, non-overlapping list of (start, end) pairs.
    """
    if not segments:
        return []
    sorted_segs = sorted(segments)
    merged: list[Segment] = [sorted_segs[0]]
    for start, end in sorted_segs[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def remove_ads(
    audio: AudioArray,
    sample_rate: int,
    strategy: Literal["timestamps", "fingerprint", "combined"] = "timestamps",
    *,
    timestamps: list[tuple[float, float]] | None = None,
    reference_clips: list[AudioArray] | None = None,
    correlation_threshold: float = 0.7,
    fade_ms: float = 20.0,
    timestamp_action: Literal["replace", "duck", "remove"] = "replace",
    timestamp_duck_db: float = -18.0,
    cut_snap_ms: float = 200.0,
    cut_match_ms: float = 30.0,
) -> AudioArray:
    """Detect and remove ads / interrupts from *audio*.

    Supports removal of advertisements, station IDs / jingles, and sponsorship
    reads.  The user can specify the segments to remove directly via *timestamps*,
    or let the fingerprint detector locate occurrences of known *reference_clips*
    automatically.

    Args:
        audio: 1-D (mono) or 2-D (samples x channels) float32 audio array.
        sample_rate: Sample rate in Hz.
        strategy: Removal strategy.  One of ``"timestamps"``, ``"fingerprint"``,
            or ``"combined"``.
        timestamps: Optional list of ``(start_s, end_s)`` pairs (in seconds)
            marking segments to remove.  Used when *strategy* is ``"timestamps"``
            or ``"combined"``; if omitted or empty, no timestamp-based removal
            is performed.
        reference_clips: Optional list of known ad/jingle audio arrays used for
            fingerprint matching.  Used when *strategy* is ``"fingerprint"`` or
            ``"combined"``; if omitted or empty, no fingerprint-based removal
            is performed.
        correlation_threshold: Minimum normalized cross-correlation score for the
            fingerprint detector to flag a match (default: 0.7).
        fade_ms: Crossfade duration at cut points in milliseconds (default: 20.0).
        timestamp_action: How timestamp-marked intervals are handled.
            ``"replace"`` replaces intervals with smooth bridges while preserving
            duration; ``"duck"`` attenuates in-place; ``"remove"`` cuts them out
            (default: ``"replace"``).
        timestamp_duck_db: Gain in dB applied to timestamp-marked intervals when
            *timestamp_action* is ``"duck"`` (default: -18.0 dB).
        cut_snap_ms: Boundary search window in milliseconds used when removing
            segments. If > 0, each cut boundary is snapped to a nearby
            low-amplitude sample to reduce splice artifacts (default: 200.0).
        cut_match_ms: Context window in milliseconds used to align post-cut
            content with pre-cut waveform when snapping remove boundaries
            (default: 30.0).

    Returns:
        Float32 audio array with detected ad segments removed.

    Example::

        import soundfile as sf
        from audio_cleaner.ads import remove_ads

        audio, sr = sf.read("podcast.flac", dtype="float32")

        # Remove known segments by timestamp
        cleaned = remove_ads(
            audio, sr,
            strategy="timestamps",
            timestamps=[(30.0, 45.0), (120.0, 135.0)],
        )
        sf.write("podcast_cleaned.flac", cleaned, sr)
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if audio.ndim not in (1, 2):
        raise ValueError(
            f"audio must be 1-D (mono) or 2-D (samples x channels), got ndim={audio.ndim}"
        )
    if audio.shape[0] == 0:
        raise ValueError("audio must not be empty")
    if fade_ms < 0:
        raise ValueError(f"fade_ms must be >= 0, got {fade_ms}")
    if not (0 <= correlation_threshold <= 1):
        raise ValueError(f"correlation_threshold must be in [0, 1], got {correlation_threshold}")
    if timestamp_action not in ("replace", "duck", "remove"):
        raise ValueError(
            "timestamp_action must be one of 'replace', 'duck', or "
            f"'remove', got {timestamp_action}"
        )
    if timestamp_action == "duck" and timestamp_duck_db > 0:
        raise ValueError(
            f"timestamp_duck_db must be <= 0 when timestamp_action='duck', got {timestamp_duck_db}"
        )
    if cut_snap_ms < 0:
        raise ValueError(f"cut_snap_ms must be >= 0, got {cut_snap_ms}")
    if cut_match_ms < 0:
        raise ValueError(f"cut_match_ms must be >= 0, got {cut_match_ms}")

    timestamp_segments: list[Segment] = []
    fingerprint_segments: list[Segment] = []

    if strategy not in ("timestamps", "fingerprint", "combined"):
        raise ValueError(
            "strategy must be one of 'timestamps', 'fingerprint', "
            f"'combined', got {strategy}"
        )

    if strategy in ("timestamps", "combined"):
        if timestamps:
            for start_s, end_s in timestamps:
                if start_s < 0:
                    raise ValueError(f"timestamp start must be >= 0, got {start_s}")
                if end_s <= start_s:
                    raise ValueError(
                        f"timestamp end ({end_s}) must be greater than start ({start_s})"
                    )
                start_sample = max(0, int(start_s * sample_rate))
                end_sample = min(audio.shape[0], int(end_s * sample_rate))
                if start_sample < end_sample:
                    timestamp_segments.append((start_sample, end_sample))

    if strategy in ("fingerprint", "combined"):
        if reference_clips:
            detector = FingerprintDetector(
                reference_clips=reference_clips,
                correlation_threshold=correlation_threshold,
            )
            fingerprint_segments.extend(detector.detect(audio, sample_rate))

    working_audio = audio.copy()
    if timestamp_segments and timestamp_action == "replace":
        merged_timestamp_segments = _merge_segments(timestamp_segments)
        working_audio = _apply_replacement(working_audio, merged_timestamp_segments)

    if timestamp_segments and timestamp_action == "duck":
        merged_timestamp_segments = _merge_segments(timestamp_segments)
        working_audio = _apply_ducking(
            working_audio,
            merged_timestamp_segments,
            sample_rate,
            duck_db=timestamp_duck_db,
            fade_ms=fade_ms,
        )

    remove_segments: list[Segment] = []
    if timestamp_segments and timestamp_action == "remove":
        remove_segments.extend(timestamp_segments)
    remove_segments.extend(fingerprint_segments)

    if not remove_segments:
        return working_audio

    merged = _merge_segments(remove_segments)
    if cut_snap_ms > 0:
        snap_samples = int(sample_rate * cut_snap_ms / 1000.0)
        match_samples = int(sample_rate * cut_match_ms / 1000.0)
        merged = _snap_remove_boundaries(working_audio, merged, snap_samples, match_samples)
    return _apply_crossfade(working_audio, merged, sample_rate, fade_ms=fade_ms)
