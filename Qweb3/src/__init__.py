"""
Qwen3-TTS Voice Cloning System - Source Module
"""

from .models import (
    VoiceProfile,
    VoiceInfo,
    GenerationRequest,
    GenerationResult,
    AudioInfo,
)
from .tts_engine import TTSEngine
from .voice_manager import VoiceManager
from .audio_utils import (
    load_audio,
    resample_audio,
    validate_reference_audio,
    save_audio,
    get_audio_info,
    get_audio_duration,
    normalize_audio,
    trim_silence,
    SUPPORTED_FORMATS,
    MIN_REFERENCE_DURATION,
    MAX_REFERENCE_DURATION,
)
from .streaming import (
    AudioStreamer,
    create_sse_event,
    stream_audio_sse,
)
from .metrics import (
    SpectralMetrics,
    ProsodyMetrics,
    QualityMetrics,
    AudioAnalysis,
    compute_spectral_metrics,
    compute_prosody_metrics,
    compute_quality_metrics,
    analyze_audio,
    compare_audio,
    format_analysis_report,
)

__version__ = "0.1.0"

__all__ = [
    # Models
    "VoiceProfile",
    "VoiceInfo",
    "GenerationRequest",
    "GenerationResult",
    "AudioInfo",
    # Engine
    "TTSEngine",
    "VoiceManager",
    # Audio utilities
    "load_audio",
    "resample_audio",
    "validate_reference_audio",
    "save_audio",
    "get_audio_info",
    "get_audio_duration",
    "normalize_audio",
    "trim_silence",
    "SUPPORTED_FORMATS",
    "MIN_REFERENCE_DURATION",
    "MAX_REFERENCE_DURATION",
    # Streaming
    "AudioStreamer",
    "create_sse_event",
    "stream_audio_sse",
    # Metrics
    "SpectralMetrics",
    "ProsodyMetrics",
    "QualityMetrics",
    "AudioAnalysis",
    "compute_spectral_metrics",
    "compute_prosody_metrics",
    "compute_quality_metrics",
    "analyze_audio",
    "compare_audio",
    "format_analysis_report",
]
