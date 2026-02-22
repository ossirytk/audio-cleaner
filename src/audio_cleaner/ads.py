"""Ad / interrupt detection and removal for FLAC and WAV audio files.

This module implements several methods for detecting and removing ads, jingles,
silence gaps, and other interrupts from audio recordings.

Methods
-------
- Silence / gap detection (RMS energy threshold)
- Loudness change-point detection (numpy.diff derivative threshold)
- Spectral dissimilarity detection (MFCC cosine distance via librosa)
- Clean segment reassembly with raised-cosine crossfade
"""

from __future__ import annotations

from dataclasses import dataclass, field
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


def _rms_db(frame: AudioArray) -> float:
    """Return the RMS level of *frame* in dBFS.

    Args:
        frame: A 1-D float32 audio frame.

    Returns:
        RMS level in dBFS. Returns -120.0 for silent (zero) frames.
    """
    rms = float(np.sqrt(np.mean(frame**2)))
    if rms < 1e-10:
        return -120.0
    return 20.0 * np.log10(rms)


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
# Detector classes
# ---------------------------------------------------------------------------


@dataclass
class SilenceDetector:
    """Detect silence / dead-air segments using per-frame RMS energy.

    Segments that are entirely below *threshold_db* for between *min_duration_s*
    and *max_duration_s* seconds are flagged for removal.

    Args:
        threshold_db: Silence threshold in dBFS (default: -45.0).
        frame_ms: Analysis frame length in milliseconds (default: 20.0).
        min_duration_s: Minimum silent segment length to remove in seconds (default: 1.0).
        max_duration_s: Maximum silent segment length to remove in seconds (default: 30.0).
    """

    threshold_db: float = -45.0
    frame_ms: float = 20.0
    min_duration_s: float = 1.0
    max_duration_s: float = 30.0

    def detect(self, audio: AudioArray, sample_rate: int) -> list[Segment]:
        """Return a list of (start, end) sample pairs that are silent.

        Args:
            audio: 1-D or 2-D float32 audio array.
            sample_rate: Sample rate in Hz.

        Returns:
            Sorted list of (start_sample, end_sample) silent segments.
        """
        mono = audio if audio.ndim == 1 else audio.mean(axis=1).astype(np.float32)
        frame_len = max(1, int(sample_rate * self.frame_ms / 1000.0))
        n_samples = mono.shape[0]
        min_frames = max(1, int(self.min_duration_s * sample_rate / frame_len))
        max_frames = int(self.max_duration_s * sample_rate / frame_len)

        silent_flags: list[bool] = []
        for i in range(0, n_samples, frame_len):
            frame = mono[i : i + frame_len]
            silent_flags.append(_rms_db(frame) < self.threshold_db)

        # Group consecutive silent frames
        segments: list[Segment] = []
        run_start: int | None = None
        for idx, is_silent in enumerate(silent_flags):
            if is_silent and run_start is None:
                run_start = idx
            elif not is_silent and run_start is not None:
                run_len = idx - run_start
                if min_frames <= run_len <= max_frames:
                    start_sample = run_start * frame_len
                    end_sample = min(idx * frame_len, n_samples)
                    segments.append((start_sample, end_sample))
                run_start = None
        if run_start is not None:
            run_len = len(silent_flags) - run_start
            if min_frames <= run_len <= max_frames:
                start_sample = run_start * frame_len
                end_sample = n_samples
                segments.append((start_sample, end_sample))

        return segments


@dataclass
class LoudnessChangeDetector:
    """Detect ad segments via abrupt loudness changes using ``numpy.diff``.

    Computes a short-term RMS loudness for each 1-second window, then uses a
    first-order difference to flag abrupt jumps.  Windows that are both loud
    (above *baseline + loudness_jump_db*) are merged into candidate segments.

    Args:
        window_s: Analysis window length in seconds (default: 1.0).
        loudness_jump_db: Minimum loudness jump in dB to flag a change point
            (default: 8.0).
        min_duration_s: Minimum segment length to remove in seconds (default: 5.0).
        max_duration_s: Maximum segment length to remove in seconds (default: 120.0).
    """

    window_s: float = 1.0
    loudness_jump_db: float = 8.0
    min_duration_s: float = 5.0
    max_duration_s: float = 120.0

    def detect(self, audio: AudioArray, sample_rate: int) -> list[Segment]:
        """Return a list of (start, end) sample pairs that are likely ad segments.

        Args:
            audio: 1-D or 2-D float32 audio array.
            sample_rate: Sample rate in Hz.

        Returns:
            Sorted list of (start_sample, end_sample) candidate segments.
        """
        mono = audio if audio.ndim == 1 else audio.mean(axis=1).astype(np.float32)
        win_len = max(1, int(sample_rate * self.window_s))
        n_samples = mono.shape[0]
        n_windows = n_samples // win_len

        if n_windows < 2:
            return []

        # Compute per-window RMS in dBFS
        loudness: list[float] = []
        for i in range(n_windows):
            frame = mono[i * win_len : (i + 1) * win_len]
            loudness.append(_rms_db(frame))

        loudness_arr = np.array(loudness, dtype=np.float64)
        baseline_db = float(np.median(loudness_arr))

        # Flag windows that are significantly louder than baseline
        loud_flags = loudness_arr > (baseline_db + self.loudness_jump_db)

        # Also flag change points (abrupt jumps) — windows immediately after a
        # large positive derivative spike
        diffs = np.diff(loudness_arr)
        jump_flags = np.zeros(n_windows, dtype=bool)
        jump_flags[1:] = diffs > self.loudness_jump_db

        candidate_flags = loud_flags | jump_flags

        # Group consecutive flagged windows into segments
        segments: list[Segment] = []
        run_start: int | None = None
        min_wins = max(1, int(self.min_duration_s / self.window_s))
        max_wins = int(self.max_duration_s / self.window_s)

        for idx, flagged in enumerate(candidate_flags):
            if flagged and run_start is None:
                run_start = idx
            elif not flagged and run_start is not None:
                run_len = idx - run_start
                if min_wins <= run_len <= max_wins:
                    segments.append((run_start * win_len, min(idx * win_len, n_samples)))
                run_start = None
        if run_start is not None:
            run_len = n_windows - run_start
            if min_wins <= run_len <= max_wins:
                segments.append((run_start * win_len, n_samples))

        return segments


@dataclass
class SpectralDissimilarityDetector:
    """Detect ad segments by comparing MFCC features to a rolling programme baseline.

    Uses ``librosa`` to compute MFCCs for each 1-second window, builds a moving
    baseline from the first *baseline_windows* windows, then flags windows whose
    cosine distance from the baseline exceeds *distance_threshold*.

    Args:
        window_s: Analysis window length in seconds (default: 1.0).
        n_mfcc: Number of MFCC coefficients (default: 13).
        baseline_windows: Number of initial windows used to build the programme
            baseline (default: 10).
        distance_threshold: Cosine distance threshold for flagging a window
            (default: 0.15).
        min_duration_s: Minimum segment length to remove in seconds (default: 5.0).
        max_duration_s: Maximum segment length to remove in seconds (default: 120.0).
    """

    window_s: float = 1.0
    n_mfcc: int = 13
    baseline_windows: int = 10
    distance_threshold: float = 0.15
    min_duration_s: float = 5.0
    max_duration_s: float = 120.0

    def detect(self, audio: AudioArray, sample_rate: int) -> list[Segment]:
        """Return a list of (start, end) sample pairs that are spectrally dissimilar.

        Args:
            audio: 1-D or 2-D float32 audio array.
            sample_rate: Sample rate in Hz.

        Returns:
            Sorted list of (start_sample, end_sample) candidate segments.
        """
        import librosa
        from scipy.spatial.distance import cosine as cosine_distance

        mono = audio if audio.ndim == 1 else audio.mean(axis=1).astype(np.float32)
        win_len = max(1, int(sample_rate * self.window_s))
        n_samples = mono.shape[0]
        n_windows = n_samples // win_len

        if n_windows < self.baseline_windows + 1:
            return []

        # Compute MFCCs once for the entire signal, then aggregate per window
        hop_length = 512
        mfcc_full = librosa.feature.mfcc(
            y=mono,
            sr=sample_rate,
            n_mfcc=self.n_mfcc,
            hop_length=hop_length,
        )
        # Number of MFCC frames that approximately span one analysis window
        frames_per_window = max(1, win_len // hop_length)
        max_mfcc_windows = mfcc_full.shape[1] // frames_per_window
        n_windows = min(n_windows, max_mfcc_windows)

        if n_windows < self.baseline_windows + 1:
            return []

        # Compute mean MFCC vector for each window by averaging frames in that window
        mfcc_vectors: list[npt.NDArray[np.float64]] = []
        for i in range(n_windows):
            start_frame = i * frames_per_window
            end_frame = start_frame + frames_per_window
            mfcc_vectors.append(mfcc_full[:, start_frame:end_frame].mean(axis=1))

        # Baseline: mean of first N windows
        baseline = np.mean(mfcc_vectors[: self.baseline_windows], axis=0)

        # Flag windows with high cosine distance from baseline
        min_wins = max(1, int(self.min_duration_s / self.window_s))
        max_wins = int(self.max_duration_s / self.window_s)

        candidate_flags = np.zeros(n_windows, dtype=bool)
        for i in range(self.baseline_windows, n_windows):
            dist = cosine_distance(mfcc_vectors[i], baseline)
            # NaN arises when a vector has zero norm (e.g., completely silent window).
            # A zero-norm MFCC vector is degenerate and indicates unusual audio content,
            # so we conservatively flag it rather than silently skip it.
            if np.isnan(dist):
                candidate_flags[i] = True
            else:
                candidate_flags[i] = dist > self.distance_threshold

        # Group consecutive flagged windows into segments
        segments: list[Segment] = []
        run_start: int | None = None
        for idx, flagged in enumerate(candidate_flags):
            if flagged and run_start is None:
                run_start = idx
            elif not flagged and run_start is not None:
                run_len = idx - run_start
                if min_wins <= run_len <= max_wins:
                    segments.append((run_start * win_len, min(idx * win_len, n_samples)))
                run_start = None
        if run_start is not None:
            run_len = n_windows - run_start
            if min_wins <= run_len <= max_wins:
                segments.append((run_start * win_len, n_samples))

        return segments


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class AdRemovalConfig:
    """Configuration for :func:`remove_ads`.

    Args:
        strategy: Detection strategy to use.  One of ``"silence"``,
            ``"loudness"``, ``"spectral"``, or ``"combined"`` (default: ``"silence"``).
        silence_threshold_db: Silence threshold in dBFS for the silence detector
            (default: -45.0).
        silence_min_duration_s: Minimum silent segment duration in seconds (default: 1.0).
        silence_max_duration_s: Maximum silent segment duration in seconds (default: 30.0).
        loudness_jump_db: Loudness jump threshold in dB for the loudness detector
            (default: 8.0).
        loudness_min_duration_s: Minimum loudness-change segment duration in seconds
            (default: 5.0).
        loudness_max_duration_s: Maximum loudness-change segment duration in seconds
            (default: 120.0).
        spectral_distance_threshold: Cosine distance threshold for spectral dissimilarity
            (default: 0.15).
        spectral_baseline_windows: Number of baseline windows for spectral detector
            (default: 10).
        spectral_min_duration_s: Minimum spectral-dissimilarity segment duration in seconds
            (default: 5.0).
        spectral_max_duration_s: Maximum spectral-dissimilarity segment duration in seconds
            (default: 120.0).
        fade_ms: Crossfade duration at cut points in milliseconds (default: 20.0).
    """

    strategy: Literal["silence", "loudness", "spectral", "combined"] = "silence"
    silence_threshold_db: float = -45.0
    silence_min_duration_s: float = 1.0
    silence_max_duration_s: float = 30.0
    loudness_jump_db: float = 8.0
    loudness_min_duration_s: float = 5.0
    loudness_max_duration_s: float = 120.0
    spectral_distance_threshold: float = 0.15
    spectral_baseline_windows: int = 10
    spectral_min_duration_s: float = 5.0
    spectral_max_duration_s: float = 120.0
    fade_ms: float = 20.0
    _silence_detector: SilenceDetector = field(init=False, repr=False)
    _loudness_detector: LoudnessChangeDetector = field(init=False, repr=False)
    _spectral_detector: SpectralDissimilarityDetector = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._silence_detector = SilenceDetector(
            threshold_db=self.silence_threshold_db,
            min_duration_s=self.silence_min_duration_s,
            max_duration_s=self.silence_max_duration_s,
        )
        self._loudness_detector = LoudnessChangeDetector(
            loudness_jump_db=self.loudness_jump_db,
            min_duration_s=self.loudness_min_duration_s,
            max_duration_s=self.loudness_max_duration_s,
        )
        self._spectral_detector = SpectralDissimilarityDetector(
            distance_threshold=self.spectral_distance_threshold,
            baseline_windows=self.spectral_baseline_windows,
            min_duration_s=self.spectral_min_duration_s,
            max_duration_s=self.spectral_max_duration_s,
        )


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
    strategy: Literal["silence", "loudness", "spectral", "combined"] = "silence",
    *,
    silence_threshold_db: float = -45.0,
    silence_min_duration_s: float = 1.0,
    silence_max_duration_s: float = 30.0,
    loudness_jump_db: float = 8.0,
    loudness_min_duration_s: float = 5.0,
    loudness_max_duration_s: float = 120.0,
    spectral_distance_threshold: float = 0.15,
    spectral_baseline_windows: int = 10,
    spectral_min_duration_s: float = 5.0,
    spectral_max_duration_s: float = 120.0,
    fade_ms: float = 20.0,
) -> AudioArray:
    """Detect and remove ads / interrupts from *audio*.

    Args:
        audio: 1-D (mono) or 2-D (samples x channels) float32 audio array.
        sample_rate: Sample rate in Hz.
        strategy: Detection strategy.  One of ``"silence"``, ``"loudness"``,
            ``"spectral"``, or ``"combined"``.
        silence_threshold_db: Silence threshold in dBFS (default: -45.0).
        silence_min_duration_s: Minimum silent segment to remove in seconds (default: 1.0).
        silence_max_duration_s: Maximum silent segment to remove in seconds (default: 30.0).
        loudness_jump_db: Loudness jump threshold in dB (default: 8.0).
        loudness_min_duration_s: Minimum loudness-change segment in seconds (default: 5.0).
        loudness_max_duration_s: Maximum loudness-change segment in seconds (default: 120.0).
        spectral_distance_threshold: Cosine distance threshold for spectral detector
            (default: 0.15).
        spectral_baseline_windows: Baseline window count for spectral detector (default: 10).
        spectral_min_duration_s: Minimum spectral-dissimilarity segment in seconds (default: 5.0).
        spectral_max_duration_s: Maximum spectral-dissimilarity segment in seconds (default: 120.0).
        fade_ms: Crossfade duration at cut points in milliseconds (default: 20.0).

    Returns:
        Float32 audio array with detected ad segments removed.

    Example::

        import soundfile as sf
        from audio_cleaner.ads import remove_ads

        audio, sr = sf.read("podcast.flac", dtype="float32")
        cleaned = remove_ads(audio, sr, strategy="silence")
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
    if silence_min_duration_s < 0:
        raise ValueError(f"silence_min_duration_s must be >= 0, got {silence_min_duration_s}")
    if silence_max_duration_s < silence_min_duration_s:
        raise ValueError(
            f"silence_max_duration_s ({silence_max_duration_s}) must be >= "
            f"silence_min_duration_s ({silence_min_duration_s})"
        )
    if loudness_min_duration_s < 0:
        raise ValueError(f"loudness_min_duration_s must be >= 0, got {loudness_min_duration_s}")
    if loudness_max_duration_s < loudness_min_duration_s:
        raise ValueError(
            f"loudness_max_duration_s ({loudness_max_duration_s}) must be >= "
            f"loudness_min_duration_s ({loudness_min_duration_s})"
        )
    if spectral_min_duration_s < 0:
        raise ValueError(f"spectral_min_duration_s must be >= 0, got {spectral_min_duration_s}")
    if spectral_max_duration_s < spectral_min_duration_s:
        raise ValueError(
            f"spectral_max_duration_s ({spectral_max_duration_s}) must be >= "
            f"spectral_min_duration_s ({spectral_min_duration_s})"
        )
    if spectral_baseline_windows < 1:
        raise ValueError(
            f"spectral_baseline_windows must be >= 1, got {spectral_baseline_windows}"
        )

    cfg = AdRemovalConfig(
        strategy=strategy,
        silence_threshold_db=silence_threshold_db,
        silence_min_duration_s=silence_min_duration_s,
        silence_max_duration_s=silence_max_duration_s,
        loudness_jump_db=loudness_jump_db,
        loudness_min_duration_s=loudness_min_duration_s,
        loudness_max_duration_s=loudness_max_duration_s,
        spectral_distance_threshold=spectral_distance_threshold,
        spectral_baseline_windows=spectral_baseline_windows,
        spectral_min_duration_s=spectral_min_duration_s,
        spectral_max_duration_s=spectral_max_duration_s,
        fade_ms=fade_ms,
    )

    segments: list[Segment] = []

    if strategy in ("silence", "combined"):
        segments.extend(cfg._silence_detector.detect(audio, sample_rate))

    if strategy in ("loudness", "combined"):
        segments.extend(cfg._loudness_detector.detect(audio, sample_rate))

    if strategy in ("spectral", "combined"):
        segments.extend(cfg._spectral_detector.detect(audio, sample_rate))

    if not segments:
        return audio.copy()

    merged = _merge_segments(segments)
    return _apply_crossfade(audio, merged, sample_rate, fade_ms=fade_ms)
