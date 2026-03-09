"""
Dependency injection for FastAPI
"""

from typing import Optional
from pathlib import Path

import config
from src import TTSEngine, VoiceManager


# Global singletons (lazy-loaded)
_engine: Optional[TTSEngine] = None
_voice_manager: Optional[VoiceManager] = None


def get_engine() -> TTSEngine:
    """
    Get the TTS engine singleton.

    The engine is created on first access but the model
    is loaded lazily on first generation.
    """
    global _engine
    if _engine is None:
        _engine = TTSEngine(
            model_name=config.MODEL_NAME,
            device=getattr(config, 'MODEL_DEVICE', None),
            dtype=config.MODEL_DTYPE,
            use_flash_attention=config.USE_FLASH_ATTENTION,
            quantization=getattr(config, 'QUANTIZATION', None),
            use_torch_compile=getattr(config, 'USE_TORCH_COMPILE', False),
        )
    return _engine


def get_voice_manager() -> VoiceManager:
    """Get the voice manager singleton."""
    global _voice_manager
    if _voice_manager is None:
        _voice_manager = VoiceManager(config.VOICES_DIR)
    return _voice_manager


def reset_singletons():
    """Reset singletons (for testing)."""
    global _engine, _voice_manager
    _engine = None
    _voice_manager = None
