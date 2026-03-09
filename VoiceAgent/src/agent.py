"""
Voice Agent Orchestrator

Connects all components for real-time voice interaction.
"""

import asyncio
import numpy as np
import sounddevice as sd
from typing import Optional
import time

import config
from .audio_capture import AudioCapture
from .stt_engine import STTEngine
from .llm_client import LLMClient
from .tts_client import TTSClient


class VoiceAgent:
    """
    Main voice agent that orchestrates the full pipeline.

    Pipeline:
    1. Audio Capture (VAD) -> Detects when user stops speaking
    2. STT (Whisper) -> Transcribes speech to text
    3. LLM (llama.cpp) -> Generates response (streaming)
    4. TTS (Qweb3) -> Converts response to speech (streaming)
    5. Audio Playback -> Plays response audio
    """

    def __init__(self):
        self.audio_capture = AudioCapture()
        self.stt = STTEngine()
        self.llm = LLMClient()
        self.tts = TTSClient()

        # Audio playback queue
        self.audio_queue = asyncio.Queue()
        self._playing = False
        self._running = False

        # Connect callback
        self.audio_capture.on_speech_end = self._on_speech_end

    async def run(self):
        """Run the voice agent."""
        self._running = True

        # Pre-load models
        print("[Agent] Loading models...")
        self.stt.load()

        # Start components
        await self.audio_capture.start()

        # Try to connect to TTS (non-fatal if fails)
        try:
            await self.tts.connect()
        except Exception as e:
            print(f"[Agent] WARNING: TTS connection failed: {e}")
            print("[Agent] Start Qweb3 TTS server: cd E:\\AgentingStuff\\tts\\Qweb3 && python main.py --api")
            print("[Agent] Continuing without TTS (text responses only)...")

        # Start playback task
        playback_task = asyncio.create_task(self._playback_loop())

        print("[Agent] Ready! Speak to interact...")
        print("[Agent] Press Ctrl+C to stop\n")

        # Keep running
        try:
            while self._running:
                await asyncio.sleep(0.1)
        finally:
            playback_task.cancel()

    async def stop(self):
        """Stop the voice agent."""
        self._running = False
        await self.audio_capture.stop()
        await self.tts.disconnect()

    async def _on_speech_end(self, audio: np.ndarray):
        """Called when user stops speaking."""
        pipeline_start = time.perf_counter()

        if config.DEBUG:
            print(f"\n[Agent] Processing speech ({len(audio)} samples)...")

        # Step 1: Transcribe
        text = self.stt.transcribe(audio)
        if not text.strip():
            print("[Agent] No speech detected")
            return

        print(f"\n[You] {text}")

        # Step 2 & 3: Generate response and stream to TTS
        print("[Agent] ", end="", flush=True)

        tts_available = self.tts._ws is not None

        async def token_generator():
            async for token in self.llm.generate_stream(text):
                print(token, end="", flush=True)
                yield token

        # Step 4: Stream audio to playback (if TTS available)
        if tts_available:
            try:
                first_audio = True
                async for audio_chunk in self.tts.stream_generate(token_generator()):
                    if first_audio:
                        first_audio = False
                        latency = (time.perf_counter() - pipeline_start) * 1000
                        if config.LOG_LATENCY:
                            print(f"\n[Latency] First audio: {latency:.0f}ms")

                    await self.audio_queue.put(audio_chunk)

                # Signal end of response
                await self.audio_queue.put(None)
            except Exception as e:
                print(f"\n[Agent] TTS error: {e}")
        else:
            # No TTS - just print the response
            async for token in token_generator():
                pass  # Already printed in generator

        print()  # New line after response

    async def _playback_loop(self):
        """Play audio chunks from the queue."""
        sample_rate = 24000  # TTS output rate

        while True:
            try:
                chunk = await self.audio_queue.get()

                if chunk is None:
                    continue

                # Convert to float32 for playback
                audio_float = chunk.astype(np.float32) / 32768.0

                # Play audio (blocking)
                sd.play(audio_float, sample_rate)
                sd.wait()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Playback] Error: {e}")
