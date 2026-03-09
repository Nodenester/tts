"""
Configuration settings for Qwen3-TTS Voice Cloning System
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.resolve()
VOICES_DIR = BASE_DIR / "voices"
OUTPUT_DIR = BASE_DIR / "output"

# Ensure directories exist
VOICES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Model settings
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
MODEL_DTYPE = "bfloat16"  # or "float16" for older GPUs
MODEL_DEVICE = "cuda:1"  # Use RTX 5060 Ti (cuda:0 is RTX 4060 used by LLM)
USE_FLASH_ATTENTION = True  # Set to False if flash-attn not installed

# Quantization settings for faster inference
# Options: None, "4bit", "8bit"
# Note: Qwen3-TTS doesn't support bitsandbytes quantization (pickle error)
QUANTIZATION = None  # Disabled - model incompatible
USE_TORCH_COMPILE = False  # Disabled - model incompatible with torch.compile

# Audio settings
SAMPLE_RATE = 24000  # Output sample rate
MIN_REFERENCE_DURATION = 3.0  # Minimum seconds for voice cloning
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]

# Server settings
API_HOST = "0.0.0.0"
API_PORT = 8000
GRADIO_PORT = 7860

# Generation settings
DEFAULT_LANGUAGE = "English"
SUPPORTED_LANGUAGES = [
    "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese",
    "Spanish", "Italian"
]

# Streaming settings
STREAM_CHUNK_SIZE = 4096  # bytes per chunk
