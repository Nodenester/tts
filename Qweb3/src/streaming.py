"""
Audio streaming utilities for Qwen3-TTS Voice Cloning System
"""

import base64
import io
import struct
from typing import Generator, Optional

import numpy as np
import soundfile as sf


class AudioStreamer:
    """
    Handle streaming audio output for real-time playback.

    Converts audio arrays to streaming chunks in WAV format.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        chunk_size: int = 4096,
        channels: int = 1,
        bits_per_sample: int = 16,
    ):
        """
        Initialize the audio streamer.

        Args:
            sample_rate: Audio sample rate
            chunk_size: Size of each chunk in bytes
            channels: Number of audio channels
            bits_per_sample: Bits per sample (16 or 32)
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.bits_per_sample = bits_per_sample
        self.bytes_per_sample = bits_per_sample // 8

    def create_wav_header(self, data_size: int) -> bytes:
        """
        Create a WAV file header.

        Args:
            data_size: Size of audio data in bytes

        Returns:
            WAV header as bytes
        """
        byte_rate = self.sample_rate * self.channels * self.bytes_per_sample
        block_align = self.channels * self.bytes_per_sample

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,  # File size - 8
            b"WAVE",
            b"fmt ",
            16,  # Subchunk1Size (16 for PCM)
            1,  # AudioFormat (1 for PCM)
            self.channels,
            self.sample_rate,
            byte_rate,
            block_align,
            self.bits_per_sample,
            b"data",
            data_size,
        )
        return header

    def float_to_pcm(self, audio: np.ndarray) -> bytes:
        """
        Convert float audio to PCM bytes.

        Args:
            audio: Audio data as float32 numpy array (-1.0 to 1.0)

        Returns:
            PCM audio data as bytes
        """
        # Clip to valid range
        audio = np.clip(audio, -1.0, 1.0)

        if self.bits_per_sample == 16:
            # Convert to 16-bit PCM
            pcm = (audio * 32767).astype(np.int16)
        elif self.bits_per_sample == 32:
            # Convert to 32-bit PCM
            pcm = (audio * 2147483647).astype(np.int32)
        else:
            raise ValueError(f"Unsupported bits_per_sample: {self.bits_per_sample}")

        return pcm.tobytes()

    def stream_audio(
        self,
        audio: np.ndarray,
        include_header: bool = True,
    ) -> Generator[bytes, None, None]:
        """
        Stream audio data as chunks.

        Args:
            audio: Audio data as numpy array
            include_header: Whether to include WAV header in first chunk

        Yields:
            Audio data chunks as bytes
        """
        # Convert to PCM
        pcm_data = self.float_to_pcm(audio)

        # Yield header first if requested
        if include_header:
            header = self.create_wav_header(len(pcm_data))
            yield header

        # Yield data chunks
        for i in range(0, len(pcm_data), self.chunk_size):
            chunk = pcm_data[i:i + self.chunk_size]
            yield chunk

    def audio_to_wav_bytes(self, audio: np.ndarray) -> bytes:
        """
        Convert audio array to complete WAV file bytes.

        Args:
            audio: Audio data as numpy array

        Returns:
            Complete WAV file as bytes
        """
        pcm_data = self.float_to_pcm(audio)
        header = self.create_wav_header(len(pcm_data))
        return header + pcm_data

    def audio_to_base64(
        self,
        audio: np.ndarray,
        format: str = "wav",
    ) -> str:
        """
        Convert audio to base64-encoded string.

        Args:
            audio: Audio data as numpy array
            format: Output format (wav, mp3, etc.)

        Returns:
            Base64-encoded audio string
        """
        buffer = io.BytesIO()

        if format == "wav":
            # Use our WAV encoder
            wav_bytes = self.audio_to_wav_bytes(audio)
            buffer.write(wav_bytes)
        else:
            # Use soundfile for other formats
            sf.write(buffer, audio, self.sample_rate, format=format)

        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    def base64_to_audio(self, b64_string: str) -> np.ndarray:
        """
        Decode base64 audio string to numpy array.

        Args:
            b64_string: Base64-encoded audio

        Returns:
            Audio data as numpy array
        """
        audio_bytes = base64.b64decode(b64_string)
        buffer = io.BytesIO(audio_bytes)
        audio, _ = sf.read(buffer, dtype="float32")
        return audio


def create_sse_event(
    data: str,
    event: Optional[str] = None,
    id: Optional[str] = None,
) -> str:
    """
    Create a Server-Sent Event formatted string.

    Args:
        data: Event data
        event: Event type
        id: Event ID

    Returns:
        SSE formatted string
    """
    lines = []

    if id:
        lines.append(f"id: {id}")
    if event:
        lines.append(f"event: {event}")

    # Data can be multi-line
    for line in data.split("\n"):
        lines.append(f"data: {line}")

    lines.append("")  # Empty line to end event
    return "\n".join(lines) + "\n"


def stream_audio_sse(
    audio: np.ndarray,
    sample_rate: int,
    chunk_duration: float = 0.5,
) -> Generator[str, None, None]:
    """
    Stream audio as Server-Sent Events with base64-encoded chunks.

    Args:
        audio: Audio data as numpy array
        sample_rate: Audio sample rate
        chunk_duration: Duration of each chunk in seconds

    Yields:
        SSE formatted events with base64 audio chunks
    """
    streamer = AudioStreamer(sample_rate=sample_rate)
    chunk_samples = int(sample_rate * chunk_duration)

    # Send start event
    yield create_sse_event('{"status": "start"}', event="tts_start")

    # Stream audio chunks
    for i, start in enumerate(range(0, len(audio), chunk_samples)):
        chunk = audio[start:start + chunk_samples]
        b64_audio = streamer.audio_to_base64(chunk)

        data = {
            "chunk_index": i,
            "audio": b64_audio,
            "sample_rate": sample_rate,
        }
        yield create_sse_event(str(data), event="audio_chunk", id=str(i))

    # Send end event
    yield create_sse_event('{"status": "complete"}', event="tts_end")
