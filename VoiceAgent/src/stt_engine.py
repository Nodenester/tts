"""
Speech-to-Text Engine using faster-whisper

Optimized for low-latency transcription with hallucination filtering.
"""

import numpy as np
import re
from typing import Optional
import time

import config


class STTEngine:
    """
    Speech-to-Text using faster-whisper.

    Provides fast transcription with GPU acceleration and hallucination filtering.
    """

    # Common Whisper hallucinations to filter out
    HALLUCINATION_PATTERNS = [
        r"^\.+$",  # Just dots
        r"^(uh+|um+|ah+|oh+|hmm+)[\.\s]*$",  # Just filler words
        r"(.{3,}?)\1{3,}",  # Repeated phrases 3+ times
        r"thank you for watching",
        r"please subscribe",
        r"like and subscribe",
        r"see you next time",
        r"goodbye",
        r"^[\s\.\,\!\?]+$",  # Just punctuation
    ]

    def __init__(self):
        self.model = None
        self.model_name = config.WHISPER_MODEL
        self.device = config.WHISPER_DEVICE
        self.compute_type = config.WHISPER_COMPUTE_TYPE
        self._hallucination_re = [re.compile(p, re.IGNORECASE) for p in self.HALLUCINATION_PATTERNS]

    def load(self):
        """Load the Whisper model."""
        from faster_whisper import WhisperModel

        print(f"[STT] Loading Whisper model '{self.model_name}' on {self.device}...")
        start = time.perf_counter()

        self.model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type
        )

        elapsed = (time.perf_counter() - start) * 1000
        print(f"[STT] Model loaded in {elapsed:.0f}ms")

    def _is_hallucination(self, text: str, audio_duration: float) -> bool:
        """Check if transcription is likely a hallucination."""
        if not text or len(text.strip()) < 2:
            return True

        # Check against known patterns
        for pattern in self._hallucination_re:
            if pattern.search(text):
                return True

        # Check for unrealistic word rate (>10 words per second is suspicious)
        words = text.split()
        if audio_duration > 0 and len(words) / audio_duration > 10:
            return True

        # Check for excessive repetition
        if len(words) > 4:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # Less than 30% unique words
                return True

        return False

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data as int16 or float32 numpy array
            sample_rate: Sample rate of the audio

        Returns:
            Transcribed text (empty string if hallucination detected)
        """
        if self.model is None:
            self.load()

        start = time.perf_counter()
        audio_duration = len(audio) / sample_rate

        # Convert to float32 if needed
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        # Check audio energy - skip if too quiet
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.01:  # Very quiet audio
            if config.DEBUG:
                print(f"[STT] Audio too quiet (RMS={rms:.4f}), skipping")
            return ""

        # Transcribe
        segments, info = self.model.transcribe(
            audio,
            language="en",
            beam_size=1,  # Faster with beam_size=1
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=True,  # Use Whisper's VAD as backup
            no_speech_threshold=0.6,
        )

        # Collect text
        text = " ".join(segment.text for segment in segments).strip()

        elapsed = (time.perf_counter() - start) * 1000

        # Check for hallucination
        if self._is_hallucination(text, audio_duration):
            if config.DEBUG:
                print(f"[STT] Filtered hallucination in {elapsed:.0f}ms: '{text[:50]}...'")
            return ""

        if config.LOG_LATENCY:
            print(f"[STT] Transcribed in {elapsed:.0f}ms: '{text[:50]}...'")

        return text
