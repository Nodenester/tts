# TTS - Text-to-Speech & Voice Agent System

> **Originally built: January 24, 2026**

A local text-to-speech system with voice cloning and a real-time voice agent, built for personal use on consumer GPUs.

## Status

**Archived** -- this was a weekend project to explore Qwen3-TTS voice cloning and wire it into a real-time voice assistant pipeline. It works but is not actively maintained.

## What's Inside

### Qweb3 -- Voice Cloning TTS Server

A complete TTS service using **Qwen3-TTS-12Hz-0.6B-Base** with voice cloning from a 3-second audio sample.

- FastAPI REST API + Gradio web UI
- Voice profile management (clone, save, load, delete)
- Real-time audio streaming (SSE + WebSocket)
- Audio quality metrics (spectral analysis, prosody, SNR)
- Docker configs for vLLM, F5-TTS, and flash-attention builds
- 10 languages supported

### VoiceAgent -- Real-Time Voice AI

A voice-activated AI agent: listen with mic, transcribe, think, speak.

- **Microphone** capture with WebRTC VAD (voice activity detection)
- **STT**: faster-whisper (39-224ms latency)
- **LLM**: llama.cpp via Docker (streaming tokens)
- **TTS**: Qweb3 server via WebSocket
- Target: sub-1s end-to-end latency (achieved ~3-4s, bottlenecked by TTS)

## Tech Stack

- Python 3.12
- Qwen3-TTS-12Hz-0.6B-Base (voice cloning model)
- FastAPI, Gradio, uvicorn
- PyTorch + FlashAttention 2 + bfloat16
- faster-whisper, sounddevice, webrtcvad
- llama.cpp (Docker, CUDA)
- httpx, websockets

## Hardware Used

- RTX 5060 Ti 16GB (TTS) + RTX 4060 8GB (LLM)
- 64GB RAM, Windows 11, CUDA 13.0

## Quick Start

```bash
# Qweb3 TTS server
cd Qweb3
pip install -r requirements.txt
python main.py --api      # REST API on :8000
python main.py --ui       # Gradio UI on :7860
python main.py --test     # Verify environment

# VoiceAgent
cd VoiceAgent
pip install -r requirements.txt
python main.py --test     # Check all components
python main.py            # Run the agent
```

## License

MIT

## Author

NodeNestor
