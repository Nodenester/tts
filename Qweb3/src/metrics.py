"""
Audio quality metrics for TTS evaluation and research.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class SpectralMetrics:
    """Spectral analysis metrics."""
    centroid_mean: float = 0.0
    centroid_std: float = 0.0
    bandwidth_mean: float = 0.0
    bandwidth_std: float = 0.0
    rolloff_mean: float = 0.0
    rolloff_std: float = 0.0
    zero_crossing_rate_mean: float = 0.0
    zero_crossing_rate_std: float = 0.0

    def to_dict(self) -> dict:
        return {
            "centroid_mean": self.centroid_mean,
            "centroid_std": self.centroid_std,
            "bandwidth_mean": self.bandwidth_mean,
            "bandwidth_std": self.bandwidth_std,
            "rolloff_mean": self.rolloff_mean,
            "rolloff_std": self.rolloff_std,
            "zero_crossing_rate_mean": self.zero_crossing_rate_mean,
            "zero_crossing_rate_std": self.zero_crossing_rate_std,
        }


@dataclass
class ProsodyMetrics:
    """Prosody and pitch metrics."""
    pitch_mean: float = 0.0
    pitch_std: float = 0.0
    pitch_min: float = 0.0
    pitch_max: float = 0.0
    pitch_range: float = 0.0
    voiced_ratio: float = 0.0
    speaking_rate_estimate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "pitch_mean": self.pitch_mean,
            "pitch_std": self.pitch_std,
            "pitch_min": self.pitch_min,
            "pitch_max": self.pitch_max,
            "pitch_range": self.pitch_range,
            "voiced_ratio": self.voiced_ratio,
            "speaking_rate_estimate": self.speaking_rate_estimate,
        }


@dataclass
class QualityMetrics:
    """Audio quality indicators."""
    rms_mean: float = 0.0
    rms_std: float = 0.0
    rms_max: float = 0.0
    dynamic_range_db: float = 0.0
    clipping_ratio: float = 0.0
    silence_ratio: float = 0.0
    snr_estimate_db: float = 0.0

    def to_dict(self) -> dict:
        return {
            "rms_mean": self.rms_mean,
            "rms_std": self.rms_std,
            "rms_max": self.rms_max,
            "dynamic_range_db": self.dynamic_range_db,
            "clipping_ratio": self.clipping_ratio,
            "silence_ratio": self.silence_ratio,
            "snr_estimate_db": self.snr_estimate_db,
        }


@dataclass
class AudioAnalysis:
    """Complete audio analysis results."""
    duration_seconds: float = 0.0
    sample_rate: int = 0
    num_samples: int = 0
    spectral: SpectralMetrics = field(default_factory=SpectralMetrics)
    prosody: ProsodyMetrics = field(default_factory=ProsodyMetrics)
    quality: QualityMetrics = field(default_factory=QualityMetrics)

    def to_dict(self) -> dict:
        return {
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "num_samples": self.num_samples,
            "spectral": self.spectral.to_dict(),
            "prosody": self.prosody.to_dict(),
            "quality": self.quality.to_dict(),
        }


def compute_spectral_metrics(audio: np.ndarray, sr: int) -> SpectralMetrics:
    """
    Compute spectral analysis metrics.

    Args:
        audio: Audio data as numpy array
        sr: Sample rate

    Returns:
        SpectralMetrics object
    """
    import librosa

    # Spectral centroid (brightness)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]

    # Spectral rolloff (85% energy cutoff)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)[0]

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)[0]

    return SpectralMetrics(
        centroid_mean=float(np.mean(centroid)),
        centroid_std=float(np.std(centroid)),
        bandwidth_mean=float(np.mean(bandwidth)),
        bandwidth_std=float(np.std(bandwidth)),
        rolloff_mean=float(np.mean(rolloff)),
        rolloff_std=float(np.std(rolloff)),
        zero_crossing_rate_mean=float(np.mean(zcr)),
        zero_crossing_rate_std=float(np.std(zcr)),
    )


def compute_prosody_metrics(audio: np.ndarray, sr: int) -> ProsodyMetrics:
    """
    Compute prosody and pitch metrics.

    Args:
        audio: Audio data as numpy array
        sr: Sample rate

    Returns:
        ProsodyMetrics object
    """
    import librosa

    # Extract pitch using pyin (probabilistic YIN)
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),  # ~65 Hz
            fmax=librosa.note_to_hz('C6'),  # ~1047 Hz
            sr=sr,
        )

        # Filter to voiced frames only
        f0_voiced = f0[~np.isnan(f0)]

        if len(f0_voiced) > 0:
            pitch_mean = float(np.mean(f0_voiced))
            pitch_std = float(np.std(f0_voiced))
            pitch_min = float(np.min(f0_voiced))
            pitch_max = float(np.max(f0_voiced))
            pitch_range = pitch_max - pitch_min
        else:
            pitch_mean = pitch_std = pitch_min = pitch_max = pitch_range = 0.0

        voiced_ratio = float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0

    except Exception:
        pitch_mean = pitch_std = pitch_min = pitch_max = pitch_range = 0.0
        voiced_ratio = 0.0

    # Estimate speaking rate (rough approximation based on energy peaks)
    try:
        # Get onset strength
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        # Estimate tempo (related to speaking rate)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0]
        speaking_rate = float(tempo / 60.0)  # Convert BPM to per-second
    except Exception:
        speaking_rate = 0.0

    return ProsodyMetrics(
        pitch_mean=pitch_mean,
        pitch_std=pitch_std,
        pitch_min=pitch_min,
        pitch_max=pitch_max,
        pitch_range=pitch_range,
        voiced_ratio=voiced_ratio,
        speaking_rate_estimate=speaking_rate,
    )


def compute_quality_metrics(audio: np.ndarray, sr: int) -> QualityMetrics:
    """
    Compute audio quality indicators.

    Args:
        audio: Audio data as numpy array
        sr: Sample rate

    Returns:
        QualityMetrics object
    """
    import librosa

    # RMS energy
    rms = librosa.feature.rms(y=audio)[0]

    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    rms_max = float(np.max(rms))

    # Dynamic range (in dB)
    rms_min_nonzero = np.min(rms[rms > 1e-10]) if np.any(rms > 1e-10) else 1e-10
    dynamic_range_db = float(20 * np.log10(rms_max / rms_min_nonzero)) if rms_max > 0 else 0.0

    # Clipping detection (samples near +/- 1.0)
    clipping_ratio = float(np.mean(np.abs(audio) > 0.99))

    # Silence detection (very low energy frames)
    silence_threshold = 0.01
    silence_ratio = float(np.mean(rms < silence_threshold))

    # Simple SNR estimate (assuming noise floor is minimum RMS)
    if rms_min_nonzero > 0 and rms_mean > rms_min_nonzero:
        snr_estimate_db = float(20 * np.log10(rms_mean / rms_min_nonzero))
    else:
        snr_estimate_db = 0.0

    return QualityMetrics(
        rms_mean=rms_mean,
        rms_std=rms_std,
        rms_max=rms_max,
        dynamic_range_db=dynamic_range_db,
        clipping_ratio=clipping_ratio,
        silence_ratio=silence_ratio,
        snr_estimate_db=snr_estimate_db,
    )


def analyze_audio(audio: np.ndarray, sr: int) -> AudioAnalysis:
    """
    Perform complete audio analysis.

    Args:
        audio: Audio data as numpy array
        sr: Sample rate

    Returns:
        AudioAnalysis object with all metrics
    """
    # Ensure mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Ensure float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    duration = len(audio) / sr

    return AudioAnalysis(
        duration_seconds=duration,
        sample_rate=sr,
        num_samples=len(audio),
        spectral=compute_spectral_metrics(audio, sr),
        prosody=compute_prosody_metrics(audio, sr),
        quality=compute_quality_metrics(audio, sr),
    )


def compare_audio(
    audio1: np.ndarray,
    sr1: int,
    audio2: np.ndarray,
    sr2: int,
) -> Dict[str, float]:
    """
    Compare two audio samples and compute similarity metrics.

    Args:
        audio1: First audio sample
        sr1: Sample rate of first audio
        audio2: Second audio sample
        sr2: Sample rate of second audio

    Returns:
        Dictionary of comparison metrics
    """
    # Analyze both
    analysis1 = analyze_audio(audio1, sr1)
    analysis2 = analyze_audio(audio2, sr2)

    # Compute normalized differences
    def safe_diff(a: float, b: float) -> float:
        """Compute relative difference."""
        if a == 0 and b == 0:
            return 0.0
        return abs(a - b) / max(abs(a), abs(b), 1e-10)

    comparison = {
        # Duration
        "duration_diff_seconds": abs(analysis1.duration_seconds - analysis2.duration_seconds),

        # Spectral differences
        "centroid_diff": safe_diff(
            analysis1.spectral.centroid_mean,
            analysis2.spectral.centroid_mean
        ),
        "bandwidth_diff": safe_diff(
            analysis1.spectral.bandwidth_mean,
            analysis2.spectral.bandwidth_mean
        ),

        # Prosody differences
        "pitch_mean_diff": safe_diff(
            analysis1.prosody.pitch_mean,
            analysis2.prosody.pitch_mean
        ),
        "pitch_std_diff": safe_diff(
            analysis1.prosody.pitch_std,
            analysis2.prosody.pitch_std
        ),

        # Quality differences
        "rms_diff": safe_diff(
            analysis1.quality.rms_mean,
            analysis2.quality.rms_mean
        ),
        "dynamic_range_diff": safe_diff(
            analysis1.quality.dynamic_range_db,
            analysis2.quality.dynamic_range_db
        ),

        # Overall similarity score (0-1, higher is more similar)
        "overall_similarity": 0.0,
    }

    # Compute overall similarity (simple average of 1 - differences)
    diffs = [
        comparison["centroid_diff"],
        comparison["bandwidth_diff"],
        comparison["pitch_mean_diff"],
        comparison["rms_diff"],
    ]
    comparison["overall_similarity"] = float(1.0 - np.mean(diffs))

    return comparison


def format_analysis_report(analysis: AudioAnalysis) -> str:
    """
    Format analysis results as a human-readable report.

    Args:
        analysis: AudioAnalysis object

    Returns:
        Formatted string report
    """
    lines = [
        "=" * 50,
        "AUDIO ANALYSIS REPORT",
        "=" * 50,
        "",
        f"Duration: {analysis.duration_seconds:.2f} seconds",
        f"Sample Rate: {analysis.sample_rate} Hz",
        f"Samples: {analysis.num_samples:,}",
        "",
        "--- Spectral Metrics ---",
        f"  Centroid (brightness): {analysis.spectral.centroid_mean:.1f} Hz (std: {analysis.spectral.centroid_std:.1f})",
        f"  Bandwidth: {analysis.spectral.bandwidth_mean:.1f} Hz (std: {analysis.spectral.bandwidth_std:.1f})",
        f"  Rolloff (85%): {analysis.spectral.rolloff_mean:.1f} Hz",
        f"  Zero Crossing Rate: {analysis.spectral.zero_crossing_rate_mean:.4f}",
        "",
        "--- Prosody Metrics ---",
        f"  Pitch Mean: {analysis.prosody.pitch_mean:.1f} Hz",
        f"  Pitch Range: {analysis.prosody.pitch_min:.1f} - {analysis.prosody.pitch_max:.1f} Hz",
        f"  Pitch Std: {analysis.prosody.pitch_std:.1f} Hz",
        f"  Voiced Ratio: {analysis.prosody.voiced_ratio:.1%}",
        f"  Speaking Rate Est.: {analysis.prosody.speaking_rate_estimate:.2f}/sec",
        "",
        "--- Quality Metrics ---",
        f"  RMS Energy: {analysis.quality.rms_mean:.4f} (max: {analysis.quality.rms_max:.4f})",
        f"  Dynamic Range: {analysis.quality.dynamic_range_db:.1f} dB",
        f"  Clipping: {analysis.quality.clipping_ratio:.2%}",
        f"  Silence Ratio: {analysis.quality.silence_ratio:.1%}",
        f"  SNR Estimate: {analysis.quality.snr_estimate_db:.1f} dB",
        "",
        "=" * 50,
    ]
    return "\n".join(lines)
