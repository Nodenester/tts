"""
Pydantic models for API request/response
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# Request Models

class CloneVoiceRequest(BaseModel):
    """Request body for voice cloning (used with form data)."""
    name: str = Field(..., description="Name for the voice profile")
    ref_text: str = Field(..., description="Transcription of the reference audio")
    language: str = Field("English", description="Language of the reference audio")


class GenerateRequest(BaseModel):
    """Request body for TTS generation."""
    text: str = Field(..., description="Text to synthesize", min_length=1)
    voice_name: str = Field(..., description="Name of the voice to use")
    language: str = Field("English", description="Language for synthesis")


class GenerateDirectRequest(BaseModel):
    """Request for direct generation without saved voice."""
    text: str = Field(..., description="Text to synthesize", min_length=1)
    ref_audio_base64: str = Field(..., description="Base64-encoded reference audio")
    ref_text: str = Field(..., description="Transcription of reference audio")
    language: str = Field("English", description="Language for synthesis")


# Response Models

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    service: str = "qwen3-tts"
    version: str = "0.1.0"


class VoiceResponse(BaseModel):
    """Voice profile information."""
    name: str
    ref_text: str
    language: str
    created_at: str


class VoiceListResponse(BaseModel):
    """List of voice profiles."""
    voices: List[VoiceResponse]
    count: int


class GenerateResponse(BaseModel):
    """TTS generation response."""
    audio_base64: str = Field(..., description="Base64-encoded WAV audio")
    sample_rate: int = Field(..., description="Audio sample rate")
    duration_seconds: float = Field(..., description="Audio duration in seconds")
    text: str = Field(..., description="Original text")
    voice_name: str = Field(..., description="Voice used for generation")


class LanguageResponse(BaseModel):
    """Supported languages."""
    languages: List[str]
    default: str = "English"


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None


class MessageResponse(BaseModel):
    """Simple message response."""
    message: str


# Metrics Models

class SpectralMetricsResponse(BaseModel):
    """Spectral analysis metrics."""
    centroid_mean: float
    centroid_std: float
    bandwidth_mean: float
    bandwidth_std: float
    rolloff_mean: float
    rolloff_std: float
    zero_crossing_rate_mean: float
    zero_crossing_rate_std: float


class ProsodyMetricsResponse(BaseModel):
    """Prosody and pitch metrics."""
    pitch_mean: float
    pitch_std: float
    pitch_min: float
    pitch_max: float
    pitch_range: float
    voiced_ratio: float
    speaking_rate_estimate: float


class QualityMetricsResponse(BaseModel):
    """Audio quality indicators."""
    rms_mean: float
    rms_std: float
    rms_max: float
    dynamic_range_db: float
    clipping_ratio: float
    silence_ratio: float
    snr_estimate_db: float


class AnalysisResponse(BaseModel):
    """Complete audio analysis response."""
    duration_seconds: float
    sample_rate: int
    num_samples: int
    spectral: SpectralMetricsResponse
    prosody: ProsodyMetricsResponse
    quality: QualityMetricsResponse


class ComparisonResponse(BaseModel):
    """Audio comparison response."""
    duration_diff_seconds: float
    centroid_diff: float
    bandwidth_diff: float
    pitch_mean_diff: float
    pitch_std_diff: float
    rms_diff: float
    dynamic_range_diff: float
    overall_similarity: float
