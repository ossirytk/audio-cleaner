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

    fade_in = _raised_cosine_fade(fade_samples)
    fade_out = fade_in[::-1].copy()

    chunks: list[AudioArray] = []
    for i, (start, end) in enumerate(kept):
        chunk: AudioArray = audio[start:end].copy()
        # Apply fade-out to all but the last chunk
        if i < len(kept) - 1:
            f_len = min(fade_samples, chunk.shape[0])
            if mono:
                chunk[-f_len:] *= fade_out[-f_len:]
            else:
                chunk[-f_len:] *= fade_out[-f_len:, np.newaxis]
        # Apply fade-in to all but the first chunk
        if i > 0:
            f_len = min(fade_samples, chunk.shape[0])
            if mono:
                chunk[:f_len] *= fade_in[:f_len]
            else:
                chunk[:f_len] *= fade_in[:f_len, np.newaxis]
        chunks.append(chunk)

    return np.concatenate(chunks, axis=0)


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

    segments: list[Segment] = []

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
                    segments.append((start_sample, end_sample))

    if strategy in ("fingerprint", "combined"):
        if reference_clips:
            detector = FingerprintDetector(
                reference_clips=reference_clips,
                correlation_threshold=correlation_threshold,
            )
            segments.extend(detector.detect(audio, sample_rate))

    if not segments:
        return audio.copy()

    merged = _merge_segments(segments)
    return _apply_crossfade(audio, merged, sample_rate, fade_ms=fade_ms)
