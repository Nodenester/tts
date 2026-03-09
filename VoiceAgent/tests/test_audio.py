"""
Audio Capture + VAD Test

Tests microphone capture and voice activity detection.
Run with: python -m tests.test_audio
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
import config
from src.audio_capture import AudioCapture


class AudioTest:
    """Test harness for audio capture and VAD."""

    def __init__(self):
        self.capture = AudioCapture()
        self.speech_count = 0
        self.total_samples = 0
        self.vad_latencies = []
        self.output_dir = Path(__file__).parent.parent / "test_output"
        self.output_dir.mkdir(exist_ok=True)

    async def on_speech_end(self, audio: np.ndarray):
        """Called when VAD detects end of speech."""
        latency_start = time.perf_counter()

        self.speech_count += 1
        self.total_samples += len(audio)

        duration_ms = len(audio) / config.SAMPLE_RATE * 1000

        print(f"\n[Speech #{self.speech_count}]")
        print(f"  Duration: {duration_ms:.0f}ms")
        print(f"  Samples: {len(audio)}")

        # Save audio to WAV file
        wav_path = self.output_dir / f"speech_{self.speech_count}.wav"
        self._save_wav(wav_path, audio)
        print(f"  Saved to: {wav_path}")

        # Record latency
        latency = (time.perf_counter() - latency_start) * 1000
        self.vad_latencies.append(latency)
        print(f"  Processing latency: {latency:.1f}ms")

    def _save_wav(self, path: Path, audio: np.ndarray):
        """Save audio as WAV file."""
        with wave.open(str(path), 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)  # 16-bit
            f.setframerate(config.SAMPLE_RATE)
            f.writeframes(audio.tobytes())

    async def run_test(self, duration: int = 30):
        """Run the audio test for specified duration."""
        print("=" * 60)
        print("Audio Capture + VAD Test")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  Sample rate: {config.SAMPLE_RATE} Hz")
        print(f"  Chunk size: {config.CHUNK_SIZE} samples")
        print(f"  VAD aggressiveness: {config.VAD_AGGRESSIVENESS}")
        print(f"  Silence threshold: {config.SILENCE_THRESHOLD_MS}ms")
        print(f"  Min speech duration: {config.MIN_SPEECH_MS}ms")
        print(f"\nTest duration: {duration} seconds")
        print(f"Output directory: {self.output_dir}")
        print("\n" + "=" * 60)
        print("Speak now! Press Ctrl+C to stop early.")
        print("=" * 60 + "\n")

        # Set callback
        self.capture.on_speech_end = self.on_speech_end

        # Start capture
        await self.capture.start()

        try:
            # Show live status
            for i in range(duration):
                status = "SPEAKING" if self.capture.is_speaking else "silent"
                print(f"\r[{i+1:2d}/{duration}s] Status: {status:8s} | Speeches: {self.speech_count}", end="", flush=True)
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n\nStopped by user")
        finally:
            await self.capture.stop()

        # Print summary
        self._print_summary()

    def _print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)

        print(f"\nTotal speech segments: {self.speech_count}")

        if self.speech_count > 0:
            total_duration = self.total_samples / config.SAMPLE_RATE
            print(f"Total speech duration: {total_duration:.1f}s")

            if self.vad_latencies:
                avg_latency = sum(self.vad_latencies) / len(self.vad_latencies)
                max_latency = max(self.vad_latencies)
                print(f"\nProcessing latencies:")
                print(f"  Average: {avg_latency:.1f}ms")
                print(f"  Max: {max_latency:.1f}ms")

            print(f"\nSaved files in: {self.output_dir}")
        else:
            print("\nNo speech detected!")
            print("Check your microphone settings.")

        print("=" * 60)


async def main():
    """Run audio test."""
    import argparse

    parser = argparse.ArgumentParser(description="Test audio capture and VAD")
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds")
    parser.add_argument("--vad", type=int, default=None, help="VAD aggressiveness (0-3)")
    parser.add_argument("--silence", type=int, default=None, help="Silence threshold in ms")
    args = parser.parse_args()

    # Override config if specified
    if args.vad is not None:
        config.VAD_AGGRESSIVENESS = args.vad
    if args.silence is not None:
        config.SILENCE_THRESHOLD_MS = args.silence

    test = AudioTest()
    await test.run_test(args.duration)


if __name__ == "__main__":
    asyncio.run(main())
