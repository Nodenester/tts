"""
Optimized Pipeline Test

Tests the voice agent with WebSocket streaming TTS.
Run with: python -m tests.test_optimized_pipeline
"""

import asyncio
import sys
import time
import wave
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import sounddevice as sd
import httpx
import config
from src.stt_engine import STTEngine
from src.tts_client import TTSClient


class MockLLM:
    """Simulates LLM streaming responses."""

    def __init__(self):
        self.responses = {
            "hello": "Hello! How can I help you today?",
            "how are you": "I'm doing great, thanks for asking!",
            "joke": "Why did the scarecrow win an award? Because he was outstanding in his field!",
            "default": "I understand. Is there anything else you'd like to know?"
        }

    async def generate_stream(self, text: str):
        """Generate a mock streaming response."""
        text_lower = text.lower()

        # Find matching response
        response = self.responses["default"]
        for key, val in self.responses.items():
            if key in text_lower:
                response = val
                break

        # Stream tokens (word by word with realistic delay)
        words = response.split()
        for i, word in enumerate(words):
            if i > 0:
                word = " " + word
            await asyncio.sleep(0.03)  # ~30ms per token like a fast LLM
            yield word


class OptimizedPipelineTest:
    """Tests the optimized pipeline with WebSocket streaming."""

    def __init__(self):
        self.stt = STTEngine()
        self.llm = MockLLM()
        self.tts = TTSClient()
        self.output_dir = Path(__file__).parent.parent / "test_output"
        self.output_dir.mkdir(exist_ok=True)

    async def generate_test_audio(self, text: str) -> np.ndarray:
        """Generate test audio using TTS HTTP API."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{config.TTS_HTTP_URL}/generate",
                json={
                    "text": text,
                    "voice_name": config.TTS_VOICE,
                    "language": config.TTS_LANGUAGE
                },
                timeout=30.0
            )

            if response.status_code != 200:
                raise Exception(f"TTS failed: {response.text}")

            import base64
            import io
            data = response.json()
            audio_bytes = base64.b64decode(data["audio_base64"])
            wav_io = io.BytesIO(audio_bytes)
            with wave.open(wav_io, 'rb') as wav_file:
                audio = np.frombuffer(wav_file.readframes(-1), dtype=np.int16)
                orig_sr = wav_file.getframerate()

            # Resample to 16kHz for Whisper
            audio_16k = self._resample(audio, orig_sr, 16000)
            return audio_16k

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling."""
        if orig_sr == target_sr:
            return audio
        duration = len(audio) / orig_sr
        target_samples = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_samples)
        resampled = np.interp(indices, np.arange(len(audio)), audio.astype(np.float32))
        return resampled.astype(np.int16)

    async def run_test(self):
        """Run optimized pipeline test."""
        print("=" * 60)
        print("Optimized Pipeline Test (WebSocket Streaming)")
        print("=" * 60)

        # Load STT model
        print("\n[1/4] Loading STT model...")
        self.stt.load()
        print("  [OK] STT model loaded")

        # Connect TTS WebSocket
        print("\n[2/4] Connecting TTS WebSocket...")
        await self.tts.connect()
        print("  [OK] TTS WebSocket connected")

        # Test scenarios
        test_inputs = [
            "Hello, how are you?",
            "Tell me a joke.",
        ]

        print("\n[3/4] Testing optimized pipeline...")

        for i, input_text in enumerate(test_inputs):
            print(f"\n  --- Test {i+1}: '{input_text}' ---")

            try:
                await self._run_single_test(input_text, i + 1)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

        # Cleanup
        await self.tts.disconnect()

        print("\n" + "=" * 60)
        print("Optimized pipeline test complete!")
        print("=" * 60)

    async def _run_single_test(self, input_text: str, test_num: int):
        """Run a single test with timing measurements."""
        pipeline_start = time.perf_counter()

        # Step 1: Generate simulated user speech
        print("  [1] Generating simulated user speech...")
        user_audio = await self.generate_test_audio(input_text)
        print(f"      Got {len(user_audio)} samples")

        # Step 2: Transcribe with STT
        print("  [2] Transcribing speech...")
        stt_start = time.perf_counter()
        transcribed = self.stt.transcribe(user_audio)
        stt_time = (time.perf_counter() - stt_start) * 1000
        print(f"      Transcribed: '{transcribed}' ({stt_time:.0f}ms)")

        # Step 3 & 4: Stream LLM response through TTS with concurrent playback
        print("  [3] Streaming LLM -> TTS -> Playback...")

        llm_start = time.perf_counter()
        first_audio_time = None
        response_tokens = []
        audio_chunks = []
        playback_started = False

        # Create token generator that tracks first token
        async def tracked_token_generator():
            nonlocal response_tokens
            async for token in self.llm.generate_stream(transcribed):
                response_tokens.append(token)
                yield token

        # Stream through TTS
        async for audio_chunk in self.tts.stream_generate(tracked_token_generator()):
            if first_audio_time is None:
                first_audio_time = time.perf_counter()
                time_to_first_audio = (first_audio_time - pipeline_start) * 1000
                print(f"\n      *** FIRST AUDIO: {time_to_first_audio:.0f}ms ***")

            audio_chunks.append(audio_chunk)

            # Start playback of first chunk immediately
            if not playback_started:
                playback_started = True
                # Play this chunk while continuing to receive
                audio_float = audio_chunk.astype(np.float32) / 32768.0
                sd.play(audio_float, 24000)

        # Wait for playback to finish
        sd.wait()

        # Play remaining chunks
        for chunk in audio_chunks[1:]:
            audio_float = chunk.astype(np.float32) / 32768.0
            sd.play(audio_float, 24000)
            sd.wait()

        # Calculate timings
        pipeline_end = time.perf_counter()
        total_time = (pipeline_end - pipeline_start) * 1000

        response_text = "".join(response_tokens)
        print(f"\n      Response: '{response_text}'")
        print(f"      Audio chunks: {len(audio_chunks)}")

        if first_audio_time:
            ttfa = (first_audio_time - pipeline_start) * 1000
            print(f"\n      Time to first audio: {ttfa:.0f}ms")
            if ttfa < 1000:
                print(f"      [PASS] Under 1 second target!")
            else:
                print(f"      [MISS] Over 1 second target")

        print(f"      Total pipeline time: {total_time:.0f}ms")

        # Save combined audio
        if audio_chunks:
            combined = np.concatenate(audio_chunks)
            wav_path = self.output_dir / f"optimized_response_{test_num}.wav"
            with wave.open(str(wav_path), 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(24000)
                f.writeframes(combined.tobytes())
            print(f"      Saved to: {wav_path}")


async def main():
    """Run optimized pipeline test."""
    test = OptimizedPipelineTest()
    await test.run_test()


if __name__ == "__main__":
    asyncio.run(main())
