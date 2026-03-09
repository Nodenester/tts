"""
Data models for Qwen3-TTS Voice Cloning System
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import json


@dataclass
class VoiceProfile:
    """Cached voice clone data for efficient generation."""
    name: str
    prompt_items: Any  # The cached voice clone prompt from qwen_tts
    ref_text: str
    language: str = "English"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary (excluding prompt_items which needs pickle)."""
        return {
            "name": self.name,
            "ref_text": self.ref_text,
            "language": self.language,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict, prompt_items: Any) -> "VoiceProfile":
        """Create from dictionary and prompt items."""
        return cls(
            name=data["name"],
            prompt_items=prompt_items,
            ref_text=data["ref_text"],
            language=data.get("language", "English"),
            created_at=data.get("created_at", datetime.now().isoformat()),
        )


@dataclass
class VoiceInfo:
    """Metadata about a saved voice profile."""
    name: str
    ref_text: str
    language: str
    created_at: str
    file_path: Path

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ref_text": self.ref_text,
            "language": self.language,
            "created_at": self.created_at,
            "file_path": str(self.file_path),
        }


@dataclass
class GenerationRequest:
    """Request parameters for TTS generation."""
    text: str
    voice_name: str
    language: str = "English"
    stream: bool = False

    def validate(self) -> bool:
        """Validate the request parameters."""
        if not self.text or not self.text.strip():
            raise ValueError("Text cannot be empty")
        if not self.voice_name or not self.voice_name.strip():
            raise ValueError("Voice name cannot be empty")
        return True


@dataclass
class GenerationResult:
    """Result of TTS generation."""
    audio_data: Any  # numpy array
    sample_rate: int
    duration_seconds: float
    voice_name: str
    text: str

    def to_dict(self) -> dict:
        """Convert to dictionary (excluding audio_data)."""
        return {
            "sample_rate": self.sample_rate,
            "duration_seconds": self.duration_seconds,
            "voice_name": self.voice_name,
            "text": self.text,
        }


@dataclass
class AudioInfo:
    """Metadata about an audio file."""
    duration_seconds: float
    sample_rate: int
    channels: int
    format: str
    file_size_bytes: int
    file_path: Optional[Path] = None

    def to_dict(self) -> dict:
        return {
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "format": self.format,
            "file_size_bytes": self.file_size_bytes,
            "file_path": str(self.file_path) if self.file_path else None,
        }
