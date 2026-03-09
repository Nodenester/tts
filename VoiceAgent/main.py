"""
VoiceAgent - Ultra-Low Latency Voice AI

Main entry point for the voice-activated AI agent.
"""

import argparse
import asyncio
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def test_environment():
    """Test that all required components are available."""
    print("=" * 60)
    print("VoiceAgent Environment Test")
    print("=" * 60)

    all_ok = True

    # Test imports
    print("\n[1/6] Testing Python imports...")

    try:
        import numpy as np
        print(f"  [OK] numpy {np.__version__}")
    except ImportError as e:
        print(f"  [FAIL] numpy: {e}")
        all_ok = False

    try:
        import sounddevice as sd
        print(f"  [OK] sounddevice {sd.__version__}")
    except ImportError as e:
        print(f"  [FAIL] sounddevice: {e}")
        all_ok = False

    try:
        import webrtcvad
        print(f"  [OK] webrtcvad")
    except ImportError as e:
        print(f"  [FAIL] webrtcvad: {e}")
        all_ok = False

    try:
        from faster_whisper import WhisperModel
        print(f"  [OK] faster-whisper")
    except ImportError as e:
        print(f"  [FAIL] faster-whisper: {e}")
        all_ok = False

    try:
        import httpx
        print(f"  [OK] httpx {httpx.__version__}")
    except ImportError as e:
        print(f"  [FAIL] httpx: {e}")
        all_ok = False

    try:
        import websockets
        print(f"  [OK] websockets {websockets.__version__}")
    except ImportError as e:
        print(f"  [FAIL] websockets: {e}")
        all_ok = False

    # Test audio devices
    print("\n[2/6] Testing audio devices...")
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_device = sd.query_devices(kind='input')
        print(f"  [OK] Default input: {input_device['name']}")
        output_device = sd.query_devices(kind='output')
        print(f"  [OK] Default output: {output_device['name']}")
    except Exception as e:
        print(f"  [FAIL] Audio devices: {e}")
        all_ok = False

    # Test CUDA for Whisper
    print("\n[3/6] Testing CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  [OK] CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  [WARN] CUDA not available, will use CPU (slower)")
    except ImportError:
        print(f"  [WARN] PyTorch not found, will use CPU for Whisper")

    # Test config
    print("\n[4/6] Testing configuration...")
    try:
        import config
        print(f"  [OK] Whisper model: {config.WHISPER_MODEL}")
        print(f"  [OK] LLM URL: {config.LLM_URL}")
        print(f"  [OK] TTS URL: {config.TTS_HTTP_URL}")
        print(f"  [OK] Target latency: {config.TARGET_LATENCY_MS}ms")
    except Exception as e:
        print(f"  [FAIL] Config: {e}")
        all_ok = False

    # Test LLM connection
    print("\n[5/6] Testing LLM connection...")
    try:
        import httpx
        import config
        response = httpx.get(f"{config.LLM_URL}/health", timeout=5.0)
        if response.status_code == 200:
            print(f"  [OK] llama.cpp server responding")
        else:
            print(f"  [WARN] llama.cpp returned status {response.status_code}")
    except httpx.ConnectError:
        print(f"  [WARN] llama.cpp not running (start with docker-compose)")
    except Exception as e:
        print(f"  [WARN] LLM connection: {e}")

    # Test TTS connection
    print("\n[6/6] Testing TTS connection...")
    try:
        import httpx
        import config
        response = httpx.get(f"{config.TTS_HTTP_URL}/", timeout=5.0)
        if response.status_code == 200:
            # Check available voices
            voices_resp = httpx.get(f"{config.TTS_HTTP_URL}/voices", timeout=5.0)
            if voices_resp.status_code == 200:
                voices = voices_resp.json().get("voices", [])
                voice_names = [v["name"] for v in voices]
                print(f"  [OK] Qweb3 TTS server responding")
                print(f"    Voices: {', '.join(voice_names) if voice_names else 'none'}")
                if config.TTS_VOICE not in voice_names:
                    print(f"  [WARN] Voice '{config.TTS_VOICE}' not found!")
            else:
                print(f"  [OK] Qweb3 TTS server responding")
        else:
            print(f"  [WARN] TTS returned status {response.status_code}")
    except httpx.ConnectError:
        print(f"  [WARN] Qweb3 TTS not running!")
        print(f"    Start it with: cd E:\\AgentingStuff\\tts\\Qweb3 && python main.py --api")
    except Exception as e:
        print(f"  [WARN] TTS: {e}")

    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("Environment test PASSED - Ready to run!")
    else:
        print("Environment test FAILED - Fix issues above")
    print("=" * 60)

    return all_ok


async def run_agent():
    """Run the voice agent."""
    from src.agent import VoiceAgent

    print("Starting VoiceAgent...")
    print("Press Ctrl+C to stop\n")

    agent = VoiceAgent()
    try:
        await agent.run()
    except KeyboardInterrupt:
        print("\nStopping agent...")
        await agent.stop()


def main():
    parser = argparse.ArgumentParser(description="VoiceAgent - Ultra-Low Latency Voice AI")
    parser.add_argument("--test", action="store_true", help="Test environment setup")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.test:
        success = test_environment()
        sys.exit(0 if success else 1)

    if args.debug:
        import config
        config.DEBUG = True

    # Run the agent
    asyncio.run(run_agent())


if __name__ == "__main__":
    main()
