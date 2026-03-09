"""
Full Pipeline Test

Tests the complete voice agent pipeline with simulated LLM.
Run with: python -m tests.test_pipeline
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


class MockLLM:
    """Simulates LLM streaming responses."""

    def __init__(self):
        self.responses = {
            "hello": "Hello! How can I help you today?",
            "how are you": "I'm doing great, thanks for asking!",
            "what time": "I don't have access to the current time, but I can help with other things.",
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

        # Stream tokens (word by word with delay)
        words = response.split()
        for i, word in enumerate(words):
            if i > 0:
                word = " " + word
            await asyncio.sleep(0.05)  # Simulate LLM token rate
            yield word


class PipelineTest:
    """Tests the full voice agent pipeline."""

    def __init__(self):
        self.stt = STTEngine()
        self.llm = MockLLM()
        self.output_dir = Path(__file__).parent.parent / "test_output"
        self.output_dir.mkdir(exist_ok=True)

    async def generate_test_audio(self, text: str) -> np.ndarray:
        """Generate test audio using TTS."""
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

    async def generate_response_audio(self, text: str) -> np.ndarray:
        """Generate response audio using TTS."""
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
                sample_rate = wav_file.getframerate()

            return audio, sample_rate

    def _save_wav(self, path: Path, audio: np.ndarray, sample_rate: int = 24000):
        """Save audio as WAV file."""
        with wave.open(str(path), 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sample_rate)
            f.writeframes(audio.tobytes())

    async def run_test(self):
        """Run pipeline test."""
        print("=" * 60)
        print("Full Pipeline Test (with Mock LLM)")
        print("=" * 60)

        # Load STT model
        print("\n[1/4] Loading STT model...")
        self.stt.load()
        print("  [OK] STT model loaded")

        # Test scenarios
        test_inputs = [
            "Hello, how are you?",
            "Tell me a joke.",
        ]

        print("\n[2/4] Testing full pipeline...")

        for i, input_text in enumerate(test_inputs):
            print(f"\n  --- Test {i+1}: '{input_text}' ---")
            pipeline_start = time.perf_counter()

            try:
                # Step 1: Generate "user speech" using TTS
                print("  [1] Generating simulated user speech...")
                user_audio = await self.generate_test_audio(input_text)
                print(f"      Got {len(user_audio)} samples")

                # Step 2: Transcribe with STT
                print("  [2] Transcribing speech...")
                stt_start = time.perf_counter()
                transcribed = self.stt.transcribe(user_audio)
                stt_time = (time.perf_counter() - stt_start) * 1000
                print(f"      Transcribed: '{transcribed}' ({stt_time:.0f}ms)")

                # Step 3: Generate LLM response
                print("  [3] Generating LLM response...")
                llm_start = time.perf_counter()
                response_tokens = []
                first_token_time = None

                async for token in self.llm.generate_stream(transcribed):
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    response_tokens.append(token)

                response_text = "".join(response_tokens)
                llm_ttft = (first_token_time - llm_start) * 1000 if first_token_time else 0
                llm_total = (time.perf_counter() - llm_start) * 1000
                print(f"      Response: '{response_text}' (TTFT: {llm_ttft:.0f}ms, Total: {llm_total:.0f}ms)")

                # Step 4: Generate TTS audio
                print("  [4] Generating TTS audio...")
                tts_start = time.perf_counter()
                response_audio, sample_rate = await self.generate_response_audio(response_text)
                tts_time = (time.perf_counter() - tts_start) * 1000
                print(f"      Got {len(response_audio)} samples ({tts_time:.0f}ms)")

                # Save response audio
                wav_path = self.output_dir / f"pipeline_response_{i+1}.wav"
                self._save_wav(wav_path, response_audio, sample_rate)
                print(f"      Saved to: {wav_path}")

                # Total pipeline time
                pipeline_time = (time.perf_counter() - pipeline_start) * 1000
                print(f"\n  Total pipeline time: {pipeline_time:.0f}ms")

                # Play audio
                print("  Playing response...")
                audio_float = response_audio.astype(np.float32) / 32768.0
                sd.play(audio_float, sample_rate)
                sd.wait()

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "=" * 60)
        print("Pipeline test complete!")
        print("=" * 60)


async def main():
    """Run pipeline test."""
    test = PipelineTest()
    await test.run_test()


if __name__ == "__main__":
    asyncio.run(main())
