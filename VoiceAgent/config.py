"""
VoiceAgent Configuration

Optimized for ultra-low latency voice interaction.
"""

from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

# =============================================================================
# Audio Settings
# =============================================================================

SAMPLE_RATE = 16000  # Whisper expects 16kHz
CHANNELS = 1
CHUNK_SIZE = 480  # 30ms at 16kHz (good for VAD)
VAD_AGGRESSIVENESS = 3  # 0-3, higher = more aggressive filtering

# Silence detection
SILENCE_THRESHOLD_MS = 500  # How long silence before we consider speech done
MIN_SPEECH_MS = 250  # Minimum speech duration to process

# =============================================================================
# Whisper STT Settings
# =============================================================================

WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large
WHISPER_DEVICE = "cuda"  # cuda or cpu
WHISPER_COMPUTE_TYPE = "float16"  # float16, int8 for speed

# =============================================================================
# LLM Settings (llama.cpp)
# =============================================================================

LLM_HOST = "localhost"
LLM_PORT = 8080
LLM_URL = f"http://{LLM_HOST}:{LLM_PORT}"

# Model settings
LLM_MAX_TOKENS = 150  # Keep responses short for speed
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.9

# System prompt for the agent
LLM_SYSTEM_PROMPT = """You are a helpful voice assistant. Keep your responses concise and conversational since they will be spoken aloud. Aim for 1-2 sentences when possible."""

# =============================================================================
# TTS Settings (Qweb3 API)
# =============================================================================

TTS_HOST = "localhost"
TTS_PORT = 8000
TTS_HTTP_URL = f"http://{TTS_HOST}:{TTS_PORT}"
TTS_WS_URL = f"ws://{TTS_HOST}:{TTS_PORT}/generate/realtime"

# TTS Backend: "edge" (fast, ~500ms) or "qweb3" (custom voice with flash-attn in Docker)
TTS_BACKEND = "qweb3"

# Edge TTS settings (when TTS_BACKEND = "edge")
EDGE_TTS_VOICE = "en-US-AriaNeural"  # Fast, natural sounding
EDGE_TTS_RATE = "+10%"

# Qweb3 TTS settings (when TTS_BACKEND = "qweb3")
TTS_VOICE = "ScarletSell"
TTS_LANGUAGE = "English"

# =============================================================================
# Latency Targets
# =============================================================================

TARGET_LATENCY_MS = 1000  # Total target: speech end to audio start

# Budget breakdown:
# - VAD detection: 50ms
# - STT transcription: 200ms
# - LLM first token: 150ms
# - TTS first audio: 100ms
# - Buffer: 500ms

# =============================================================================
# Debug Settings
# =============================================================================

DEBUG = True
LOG_LATENCY = True  # Log timing for each component
