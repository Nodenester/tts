"""
Audio processing utilities for Qwen3-TTS Voice Cloning System
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import soundfile as sf

from .models import AudioInfo


# Supported audio formats
SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}

# Default constraints
MIN_REFERENCE_DURATION = 3.0  # seconds
MAX_REFERENCE_DURATION = 30.0  # seconds
DEFAULT_SAMPLE_RATE = 24000


def load_audio(
    path: Union[str, Path],
    target_sr: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Load audio from file.

    Args:
        path: Path to audio file
        target_sr: Target sample rate (resamples if different)

    Returns:
        Tuple of (audio_data, sample_rate)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format not supported
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported audio format: {suffix}. Supported: {SUPPORTED_FORMATS}")

    # Load audio
    audio, sr = sf.read(str(path), dtype="float32")

    # Convert stereo to mono if needed
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Resample if needed
    if target_sr is not None and sr != target_sr:
        audio = resample_audio(audio, sr, target_sr)
        sr = target_sr

    return audio, sr


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """
    Resample audio to target sample rate.

    Args:
        audio: Audio data as numpy array
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio data
    """
    if orig_sr == target_sr:
        return audio

    # Use librosa for resampling
    import librosa
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def validate_reference_audio(
    audio: np.ndarray,
    sr: int,
    min_duration: float = MIN_REFERENCE_DURATION,
    max_duration: float = MAX_REFERENCE_DURATION,
) -> Tuple[bool, Optional[str]]:
    """
    Validate audio for use as voice cloning reference.

    Args:
        audio: Audio data as numpy array
        sr: Sample rate
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds

    Returns:
        Tuple of (is_valid, error_message)
    """
    duration = len(audio) / sr

    if duration < min_duration:
        return False, f"Audio too short: {duration:.1f}s (minimum: {min_duration}s)"

    if duration > max_duration:
        return False, f"Audio too long: {duration:.1f}s (maximum: {max_duration}s)"

    # Check for silence or very low volume
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 0.001:
        return False, "Audio appears to be silent or very quiet"

    # Check for clipping
    if np.max(np.abs(audio)) > 0.99:
        # This is a warning, not an error
        pass

    return True, None


def save_audio(
    audio: np.ndarray,
    sr: int,
    path: Union[str, Path],
    format: str = "wav",
) -> Path:
    """
    Save audio to file.

    Args:
        audio: Audio data as numpy array
        sr: Sample rate
        path: Output path
        format: Output format (wav, flac, ogg)

    Returns:
        Path to saved file
    """
    path = Path(path)

    # Ensure correct extension
    if not path.suffix:
        path = path.with_suffix(f".{format}")

    # Create parent directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize audio to prevent clipping
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val

    sf.write(str(path), audio, sr)
    return path


def get_audio_info(path: Union[str, Path]) -> AudioInfo:
    """
    Get information about an audio file.

    Args:
        path: Path to audio file

    Returns:
        AudioInfo object with file metadata
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    info = sf.info(str(path))
    file_size = os.path.getsize(path)

    return AudioInfo(
        duration_seconds=info.duration,
        sample_rate=info.samplerate,
        channels=info.channels,
        format=info.format,
        file_size_bytes=file_size,
        file_path=path,
    )


def get_audio_duration(audio: np.ndarray, sr: int) -> float:
    """Get duration of audio in seconds."""
    return len(audio) / sr


def normalize_audio(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """
    Normalize audio to target dB level.

    Args:
        audio: Audio data
        target_db: Target peak level in dB

    Returns:
        Normalized audio
    """
    # Calculate current peak in dB
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio

    current_db = 20 * np.log10(peak)
    gain_db = target_db - current_db
    gain = 10 ** (gain_db / 20)

    return audio * gain


def trim_silence(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -40.0,
    min_silence_duration: float = 0.1,
) -> np.ndarray:
    """
    Trim leading and trailing silence from audio.

    Args:
        audio: Audio data
        sr: Sample rate
        threshold_db: Silence threshold in dB
        min_silence_duration: Minimum silence duration to trim (seconds)

    Returns:
        Trimmed audio
    """
    import librosa

    # Use librosa's trim function
    trimmed, _ = librosa.effects.trim(
        audio,
        top_db=-threshold_db,
        frame_length=int(sr * min_silence_duration),
    )
    return trimmed
