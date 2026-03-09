"""
STT Engine Test

Tests faster-whisper transcription latency and accuracy.
Run with: python -m tests.test_stt
"""

import asyncio
import sys
import time
from pathlib import Path
import wave

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import httpx
import config
from src.stt_engine import STTEngine


class STTTest:
    """Test harness for STT engine."""

    def __init__(self):
        self.stt = STTEngine()
        self.output_dir = Path(__file__).parent.parent / "test_output"
        self.output_dir.mkdir(exist_ok=True)

    async def generate_test_audio(self, text: str) -> np.ndarray:
        """Generate test audio using Qweb3 TTS."""
        print(f"  Generating audio for: '{text}'")

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

            data = response.json()
            import base64
            audio_bytes = base64.b64decode(data["audio_base64"])

            # The audio is a WAV file, need to parse it
            import io
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

        # Use linear interpolation
        duration = len(audio) / orig_sr
        target_samples = int(duration * target_sr)

        indices = np.linspace(0, len(audio) - 1, target_samples)
        resampled = np.interp(indices, np.arange(len(audio)), audio.astype(np.float32))

        return resampled.astype(np.int16)

    def _save_wav(self, path: Path, audio: np.ndarray, sample_rate: int = 16000):
        """Save audio as WAV file."""
        with wave.open(str(path), 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sample_rate)
            f.writeframes(audio.tobytes())

    async def run_test(self):
        """Run STT tests."""
        print("=" * 60)
        print("STT Engine Test (faster-whisper)")
        print("=" * 60)

        # Load model
        print(f"\n[1/4] Loading Whisper model '{config.WHISPER_MODEL}'...")
        load_start = time.perf_counter()
        self.stt.load()
        load_time = (time.perf_counter() - load_start) * 1000
        print(f"  Model loaded in {load_time:.0f}ms")

        # Test phrases
        test_phrases = [
            "Hello, how are you today?",
            "What time is it?",
            "Tell me a joke.",
            "The quick brown fox jumps over the lazy dog.",
        ]

        print(f"\n[2/4] Generating test audio...")
        results = []

        for i, phrase in enumerate(test_phrases):
            print(f"\n  Test {i+1}/{len(test_phrases)}: '{phrase}'")

            try:
                # Generate audio
                audio = await self.generate_test_audio(phrase)
                duration_ms = len(audio) / 16000 * 1000

                # Save for debugging
                wav_path = self.output_dir / f"stt_test_{i+1}.wav"
                self._save_wav(wav_path, audio)

                # Transcribe
                transcribe_start = time.perf_counter()
                transcribed = self.stt.transcribe(audio)
                transcribe_time = (time.perf_counter() - transcribe_start) * 1000

                results.append({
                    "original": phrase,
                    "transcribed": transcribed,
                    "audio_ms": duration_ms,
                    "transcribe_ms": transcribe_time,
                    "rtf": transcribe_time / duration_ms  # Real-time factor
                })

                print(f"    Audio: {duration_ms:.0f}ms")
                print(f"    Transcribed: '{transcribed}'")
                print(f"    Latency: {transcribe_time:.0f}ms (RTF: {transcribe_time/duration_ms:.2f}x)")

            except Exception as e:
                print(f"    ERROR: {e}")
                results.append({
                    "original": phrase,
                    "error": str(e)
                })

        # Print summary
        self._print_summary(results)

    def _print_summary(self, results):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)

        successful = [r for r in results if "error" not in r]

        if successful:
            avg_latency = sum(r["transcribe_ms"] for r in successful) / len(successful)
            avg_rtf = sum(r["rtf"] for r in successful) / len(successful)
            max_latency = max(r["transcribe_ms"] for r in successful)

            print(f"\nResults ({len(successful)}/{len(results)} successful):")
            print(f"  Average latency: {avg_latency:.0f}ms")
            print(f"  Max latency: {max_latency:.0f}ms")
            print(f"  Average RTF: {avg_rtf:.2f}x")

            if avg_latency < 200:
                print(f"\n  [PASS] Latency target (<200ms) achieved!")
            else:
                print(f"\n  [WARN] Latency target (<200ms) not met")

            # Accuracy check
            print("\n  Transcriptions:")
            for r in successful:
                original = r["original"].lower().strip()
                transcribed = r["transcribed"].lower().strip()
                match = "OK" if original in transcribed or transcribed in original else "~"
                print(f"    [{match}] '{r['original']}' -> '{r['transcribed']}'")
        else:
            print("\n  All tests failed!")

        print("=" * 60)


async def main():
    """Run STT test."""
    test = STTTest()
    await test.run_test()


if __name__ == "__main__":
    asyncio.run(main())
