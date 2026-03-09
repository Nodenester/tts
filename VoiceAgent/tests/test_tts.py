"""
TTS Client Test

Tests Qweb3 WebSocket streaming.
Run with: python -m tests.test_tts
"""

import asyncio
import sys
import time
import json
import base64
import wave
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import websockets
import httpx
import config


class TTSTest:
    """Test harness for TTS WebSocket client."""

    def __init__(self):
        self.output_dir = Path(__file__).parent.parent / "test_output"
        self.output_dir.mkdir(exist_ok=True)

    def _save_wav(self, path: Path, audio: np.ndarray, sample_rate: int = 24000):
        """Save audio as WAV file."""
        with wave.open(str(path), 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sample_rate)
            f.writeframes(audio.tobytes())

    async def run_test(self):
        """Run TTS tests."""
        print("=" * 60)
        print("TTS Client Test (Qweb3 WebSocket)")
        print("=" * 60)

        # Check HTTP health first
        print(f"\n[1/4] Checking TTS server at {config.TTS_HTTP_URL}...")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{config.TTS_HTTP_URL}/", timeout=5.0)
                if response.status_code == 200:
                    print("  [OK] TTS HTTP server responding")
                else:
                    print(f"  [FAIL] TTS returned status {response.status_code}")
                    return
        except Exception as e:
            print(f"  [FAIL] TTS not responding: {e}")
            return

        # Test WebSocket connection
        print(f"\n[2/4] Testing WebSocket at {config.TTS_WS_URL}...")

        try:
            async with websockets.connect(config.TTS_WS_URL) as ws:
                # Send init
                init_msg = {
                    "type": "init",
                    "voice_name": config.TTS_VOICE,
                    "language": config.TTS_LANGUAGE
                }
                await ws.send(json.dumps(init_msg))

                # Wait for ready
                response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                data = json.loads(response)

                if data.get("type") == "ready":
                    print(f"  [OK] WebSocket connected, session: {data.get('session_id', 'N/A')}")
                else:
                    print(f"  [FAIL] Unexpected response: {data}")
                    return

        except Exception as e:
            print(f"  [FAIL] WebSocket error: {e}")
            return

        # Test streaming
        print(f"\n[3/4] Testing token streaming...")

        test_texts = [
            "Hello there!",
            "The weather is nice today.",
        ]

        results = []

        for i, text in enumerate(test_texts):
            print(f"\n  Test {i+1}/{len(test_texts)}: '{text}'")

            try:
                result = await self._test_streaming(text)
                results.append(result)

                print(f"    First audio: {result['ttfa_ms']:.0f}ms")
                print(f"    Total time: {result['total_ms']:.0f}ms")
                print(f"    Audio chunks: {result['chunks']}")
                print(f"    Audio samples: {result['samples']}")

                # Save audio
                if result['audio'] is not None:
                    wav_path = self.output_dir / f"tts_test_{i+1}.wav"
                    self._save_wav(wav_path, result['audio'])
                    print(f"    Saved to: {wav_path}")

            except Exception as e:
                print(f"    ERROR: {e}")
                results.append({"error": str(e)})

        # Summary
        self._print_summary(results)

    async def _test_streaming(self, text: str) -> dict:
        """Test streaming a text phrase."""
        start = time.perf_counter()
        first_audio_time = None
        audio_chunks = []

        async with websockets.connect(config.TTS_WS_URL) as ws:
            # Init
            await ws.send(json.dumps({
                "type": "init",
                "voice_name": config.TTS_VOICE,
                "language": config.TTS_LANGUAGE
            }))

            # Wait for ready
            response = await ws.recv()
            data = json.loads(response)
            if data.get("type") != "ready":
                raise Exception(f"Init failed: {data}")

            # Stream tokens (simulate word-by-word)
            tokens = text.split()
            for j, token in enumerate(tokens):
                # Add space between words
                if j > 0:
                    token = " " + token

                await ws.send(json.dumps({"type": "token", "token": token}))
                await asyncio.sleep(0.05)  # Simulate LLM token rate

            # Signal end
            await ws.send(json.dumps({"type": "end"}))

            # Receive audio chunks
            while True:
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    data = json.loads(response)

                    if data.get("type") == "audio":
                        if first_audio_time is None:
                            first_audio_time = time.perf_counter()

                        audio_bytes = base64.b64decode(data["audio_base64"])
                        # Parse WAV to get raw audio
                        import io
                        wav_io = io.BytesIO(audio_bytes)
                        with wave.open(wav_io, 'rb') as wav_file:
                            audio_chunk = np.frombuffer(wav_file.readframes(-1), dtype=np.int16)
                        audio_chunks.append(audio_chunk)

                    elif data.get("type") == "done":
                        break

                    elif data.get("type") == "error":
                        raise Exception(data.get("message", "Unknown error"))

                except asyncio.TimeoutError:
                    break

        end_time = time.perf_counter()

        # Combine audio
        audio = np.concatenate(audio_chunks) if audio_chunks else None

        return {
            "text": text,
            "ttfa_ms": (first_audio_time - start) * 1000 if first_audio_time else 0,
            "total_ms": (end_time - start) * 1000,
            "chunks": len(audio_chunks),
            "samples": len(audio) if audio is not None else 0,
            "audio": audio
        }

    def _print_summary(self, results):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)

        successful = [r for r in results if "error" not in r]

        if successful:
            avg_ttfa = sum(r["ttfa_ms"] for r in successful) / len(successful)
            avg_total = sum(r["total_ms"] for r in successful) / len(successful)

            print(f"\nResults ({len(successful)}/{len(results)} successful):")
            print(f"  Average TTFA: {avg_ttfa:.0f}ms")
            print(f"  Average total: {avg_total:.0f}ms")

            if avg_ttfa < 500:  # Relaxed target for testing
                print(f"\n  [PASS] TTS streaming working!")
            else:
                print(f"\n  [WARN] TTFA may be high")
        else:
            print("\n  All tests failed!")

        print("=" * 60)


async def main():
    """Run TTS test."""
    test = TTSTest()
    await test.run_test()


if __name__ == "__main__":
    asyncio.run(main())
