"""
Audio Capture with Voice Activity Detection

Captures microphone input and detects when user stops speaking.
"""

import numpy as np
import sounddevice as sd
import webrtcvad
from collections import deque
from typing import Optional, Callable
import asyncio
import queue
import threading

import config


class AudioCapture:
    """
    Captures audio from microphone with voice activity detection.

    Optimized for low-latency speech endpoint detection.
    """

    def __init__(self):
        self.sample_rate = config.SAMPLE_RATE
        self.channels = config.CHANNELS
        self.chunk_size = config.CHUNK_SIZE
        self.vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)

        # Audio buffer
        self.audio_buffer = deque(maxlen=int(self.sample_rate * 30))  # 30 seconds max
        self.speech_buffer = []

        # State
        self.is_speaking = False
        self.silence_frames = 0
        self.speech_frames = 0

        # Callbacks
        self.on_speech_end: Optional[Callable] = None

        # Stream
        self._stream = None
        self._running = False

        # Thread-safe queue for speech data
        self._speech_queue = queue.Queue()
        self._event_loop = None
        self._process_task = None

    def _audio_callback(self, indata, frames, time_info, status):
        """Called for each audio chunk from the microphone."""
        if status:
            print(f"Audio status: {status}")

        # Convert to 16-bit PCM for VAD
        audio_float = indata[:, 0]
        audio_int16 = (audio_float * 32767).astype(np.int16)

        # Check energy level first - ignore very quiet audio
        rms = np.sqrt(np.mean(audio_float ** 2))
        if rms < 0.005:  # Energy threshold
            is_speech = False
        else:
            # Check voice activity
            try:
                is_speech = self.vad.is_speech(audio_int16.tobytes(), self.sample_rate)
            except Exception:
                is_speech = False

        if is_speech:
            self.speech_frames += 1
            self.silence_frames = 0
            self.speech_buffer.extend(audio_int16)

            if not self.is_speaking:
                min_frames = int(config.MIN_SPEECH_MS / (self.chunk_size / self.sample_rate * 1000))
                if self.speech_frames >= min_frames:
                    self.is_speaking = True
                    if config.DEBUG:
                        print("[VAD] Speech started")
        else:
            self.silence_frames += 1

            if self.is_speaking:
                # Still accumulate audio during brief pauses
                self.speech_buffer.extend(audio_int16)

                # Check if silence threshold exceeded
                silence_ms = (self.silence_frames * self.chunk_size / self.sample_rate) * 1000
                if silence_ms >= config.SILENCE_THRESHOLD_MS:
                    self.is_speaking = False
                    self.speech_frames = 0

                    if config.DEBUG:
                        print(f"[VAD] Speech ended ({len(self.speech_buffer)} samples)")

                    # Put audio in queue for async processing
                    if len(self.speech_buffer) > 0:
                        audio_data = np.array(self.speech_buffer, dtype=np.int16)
                        self.speech_buffer = []
                        self._speech_queue.put(audio_data)

    async def _process_speech_queue(self):
        """Process speech data from the queue."""
        while self._running:
            try:
                # Check queue with timeout
                try:
                    audio_data = self._speech_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.01)  # 10ms poll
                    continue

                # Call the callback
                if self.on_speech_end:
                    await self.on_speech_end(audio_data)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[AudioCapture] Error processing speech: {e}")

    async def start(self):
        """Start capturing audio."""
        self._running = True
        self._event_loop = asyncio.get_running_loop()

        # Start the queue processor
        self._process_task = asyncio.create_task(self._process_speech_queue())

        # Start audio stream
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.chunk_size,
            dtype=np.float32,
            callback=self._audio_callback
        )
        self._stream.start()
        print(f"[AudioCapture] Started (sample_rate={self.sample_rate}, chunk={self.chunk_size})")

    async def stop(self):
        """Stop capturing audio."""
        self._running = False

        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        print("[AudioCapture] Stopped")
