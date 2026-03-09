"""
Voice profile management for Qwen3-TTS Voice Cloning System
"""

import json
import pickle
from pathlib import Path
from typing import List, Optional

from .models import VoiceProfile, VoiceInfo


class VoiceManager:
    """Manages saved voice profiles on disk."""

    def __init__(self, voices_dir: Path | str):
        """
        Initialize the voice manager.

        Args:
            voices_dir: Directory to store voice profiles
        """
        self.voices_dir = Path(voices_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)

    def _get_voice_path(self, name: str) -> Path:
        """Get the path for a voice profile."""
        # Sanitize name for filesystem
        safe_name = "".join(c for c in name if c.isalnum() or c in "._- ").strip()
        return self.voices_dir / f"{safe_name}.voice"

    def _get_metadata_path(self, name: str) -> Path:
        """Get the path for voice metadata."""
        safe_name = "".join(c for c in name if c.isalnum() or c in "._- ").strip()
        return self.voices_dir / f"{safe_name}.json"

    def save_voice(self, voice_profile: VoiceProfile) -> Path:
        """
        Save a voice profile to disk.

        Args:
            voice_profile: The voice profile to save

        Returns:
            Path to the saved voice file
        """
        voice_path = self._get_voice_path(voice_profile.name)
        metadata_path = self._get_metadata_path(voice_profile.name)

        # Save the prompt items (pickle for complex objects)
        with open(voice_path, "wb") as f:
            pickle.dump(voice_profile.prompt_items, f)

        # Save metadata as JSON
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(voice_profile.to_dict(), f, indent=2, ensure_ascii=False)

        return voice_path

    def load_voice(self, name: str) -> Optional[VoiceProfile]:
        """
        Load a voice profile from disk.

        Args:
            name: Name of the voice to load

        Returns:
            VoiceProfile if found, None otherwise
        """
        voice_path = self._get_voice_path(name)
        metadata_path = self._get_metadata_path(name)

        if not voice_path.exists() or not metadata_path.exists():
            return None

        # Load prompt items
        with open(voice_path, "rb") as f:
            prompt_items = pickle.load(f)

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return VoiceProfile.from_dict(metadata, prompt_items)

    def list_voices(self) -> List[VoiceInfo]:
        """
        List all available voice profiles.

        Returns:
            List of VoiceInfo objects
        """
        voices = []

        for metadata_path in self.voices_dir.glob("*.json"):
            voice_path = metadata_path.with_suffix(".voice")
            if not voice_path.exists():
                continue

            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                voices.append(VoiceInfo(
                    name=metadata["name"],
                    ref_text=metadata["ref_text"],
                    language=metadata.get("language", "English"),
                    created_at=metadata.get("created_at", "unknown"),
                    file_path=voice_path,
                ))
            except (json.JSONDecodeError, KeyError):
                # Skip invalid files
                continue

        return sorted(voices, key=lambda v: v.created_at, reverse=True)

    def delete_voice(self, name: str) -> bool:
        """
        Delete a voice profile.

        Args:
            name: Name of the voice to delete

        Returns:
            True if deleted, False if not found
        """
        voice_path = self._get_voice_path(name)
        metadata_path = self._get_metadata_path(name)

        deleted = False

        if voice_path.exists():
            voice_path.unlink()
            deleted = True

        if metadata_path.exists():
            metadata_path.unlink()
            deleted = True

        return deleted

    def voice_exists(self, name: str) -> bool:
        """Check if a voice profile exists."""
        voice_path = self._get_voice_path(name)
        metadata_path = self._get_metadata_path(name)
        return voice_path.exists() and metadata_path.exists()
