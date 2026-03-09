"""
Fast Qwen3-TTS server with flash-attention for VoiceAgent.

Supports:
- WebSocket realtime streaming (init, token, end protocol)
- HTTP /generate endpoint
- Voice profile loading from disk
"""
import asyncio
import base64
import io
import json
import os
import pickle
import re
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

app = FastAPI(title="Qwen3-TTS Flash Server")

# Global model instance
model = None
voice_profiles: Dict[str, Any] = {}

MODEL_NAME = os.getenv("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
SAMPLE_RATE = 24000
VOICES_DIR = Path("/app/voices")


class TTSRequest(BaseModel):
    text: str
    voice_name: Optional[str] = None
    language: str = "English"


class GenerateResponse(BaseModel):
    audio_base64: str
    sample_rate: int
    duration: float


def load_model():
    """Load the TTS model with flash attention."""
    global model
    print(f"Loading model: {MODEL_NAME}")

    from qwen_tts import Qwen3TTSModel

    # Determine dtype
    torch_dtype = torch.bfloat16

    # Check for attention implementation
    attn_impl = None
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
        print("Using flash_attention_2!")
    except ImportError:
        # Fall back to SDPA
        attn_impl = "sdpa"
        print("Flash-attn not available, using SDPA (PyTorch scaled_dot_product_attention)")

    # Load model
    load_kwargs = {
        "device_map": "cuda:0",
        "dtype": torch_dtype,
    }
    if attn_impl:
        load_kwargs["attn_implementation"] = attn_impl

    model = Qwen3TTSModel.from_pretrained(MODEL_NAME, **load_kwargs)
    print("Model loaded successfully!")

    return model


def load_voice_profiles():
    """Load all voice profiles from the voices directory."""
    global voice_profiles

    if not VOICES_DIR.exists():
        print(f"Voices directory not found: {VOICES_DIR}")
        return

    for json_path in VOICES_DIR.glob("*.json"):
        voice_path = json_path.with_suffix(".voice")
        if not voice_path.exists():
            continue

        try:
            # Load metadata
            with open(json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # Load pickled prompt items
            with open(voice_path, "rb") as f:
                prompt_items = pickle.load(f)

            name = metadata["name"]
            voice_profiles[name] = {
                "prompt_items": prompt_items,
                "ref_text": metadata["ref_text"],
                "language": metadata.get("language", "English"),
            }
            print(f"Loaded voice profile: {name}")

        except Exception as e:
            print(f"Failed to load voice {json_path.stem}: {e}")
            traceback.print_exc()

    print(f"Loaded {len(voice_profiles)} voice profiles")


def audio_to_base64(audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Convert audio numpy array to base64 WAV."""
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def generate_audio(text: str, voice_name: Optional[str], language: str = "English") -> tuple:
    """Generate audio for text using a voice profile."""
    if model is None:
        raise RuntimeError("Model not loaded")

    voice = voice_profiles.get(voice_name) if voice_name else None

    if voice:
        # Use cached voice profile
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=voice["prompt_items"],
        )
    else:
        # Simple TTS without voice cloning
        wavs, sr = model.generate_voice_design(
            text=text,
            language=language,
            instruct="A natural, clear speaking voice",
        )

    audio_data = wavs[0] if isinstance(wavs, list) else wavs
    return audio_data, sr


@app.on_event("startup")
async def startup():
    """Load model and voices on startup."""
    load_model()
    load_voice_profiles()


@app.get("/")
async def root():
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "loaded": model is not None}


@app.get("/voices")
async def list_voices():
    """List available voice profiles."""
    voices = [
        {"name": name, "language": info.get("language", "English")}
        for name, info in voice_profiles.items()
    ]
    return {"voices": voices, "count": len(voices)}


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: TTSRequest):
    """Generate speech from text."""
    try:
        start = time.perf_counter()

        audio_data, sr = generate_audio(
            text=request.text,
            voice_name=request.voice_name,
            language=request.language,
        )

        elapsed = time.perf_counter() - start
        duration = len(audio_data) / sr

        print(f"Generated {duration:.2f}s audio in {elapsed:.2f}s (RTF: {elapsed/duration:.2f})")

        return GenerateResponse(
            audio_base64=audio_to_base64(audio_data, sr),
            sample_rate=sr,
            duration=duration,
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Realtime TTS Session
class RealtimeTTSSession:
    """Manages a realtime TTS streaming session."""

    def __init__(self, voice_name: Optional[str], language: str):
        self.voice_name = voice_name
        self.language = language
        self.text_buffer = ""
        # Sentence-ending punctuation
        self.sentence_endings = re.compile(r'[.!?;]\s*')
        # Clause-ending punctuation
        self.clause_endings = re.compile(r'[,:\-]\s*')
        self.min_chunk_length = 10

    def add_token(self, token: str) -> Optional[str]:
        """Add a token and return complete sentence if ready."""
        self.text_buffer += token

        # Check for sentence endings
        if self.sentence_endings.search(self.text_buffer):
            sentence = self.text_buffer.strip()
            self.text_buffer = ""
            return sentence

        # Check for clause endings (only if buffer is long enough)
        if len(self.text_buffer) >= self.min_chunk_length:
            if self.clause_endings.search(self.text_buffer):
                clause = self.text_buffer.strip()
                self.text_buffer = ""
                return clause

        return None

    def flush(self) -> Optional[str]:
        """Flush remaining text buffer."""
        if self.text_buffer.strip():
            text = self.text_buffer.strip()
            self.text_buffer = ""
            return text
        return None


@app.websocket("/generate/realtime")
async def realtime_tts_stream(websocket: WebSocket):
    """
    Bidirectional WebSocket for realtime TTS streaming.

    Protocol:
    - {"type": "init", "voice_name": "...", "language": "English"}
    - {"type": "token", "token": "Hello"}
    - {"type": "end"}

    Responses:
    - {"type": "ready"}
    - {"type": "audio", "audio_base64": "...", "sample_rate": 24000, "duration": 1.5}
    - {"type": "done"}
    - {"type": "error", "error": "..."}
    """
    await websocket.accept()

    session = None

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            msg_type = msg.get("type", "")

            if msg_type == "init":
                voice_name = msg.get("voice_name")
                language = msg.get("language", "English")

                # Check if voice exists
                if voice_name and voice_name not in voice_profiles:
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Voice '{voice_name}' not found. Available: {list(voice_profiles.keys())}"
                    })
                    continue

                session = RealtimeTTSSession(voice_name, language)
                await websocket.send_json({"type": "ready"})

            elif msg_type == "token":
                if session is None:
                    await websocket.send_json({
                        "type": "error",
                        "error": "Session not initialized. Send init first."
                    })
                    continue

                token = msg.get("token", "")
                complete_text = session.add_token(token)

                if complete_text:
                    try:
                        start = time.perf_counter()
                        audio_data, sr = generate_audio(
                            text=complete_text,
                            voice_name=session.voice_name,
                            language=session.language,
                        )
                        elapsed = time.perf_counter() - start
                        duration = len(audio_data) / sr
                        print(f"Generated '{complete_text[:30]}...' ({duration:.2f}s) in {elapsed:.2f}s")

                        await websocket.send_json({
                            "type": "audio",
                            "audio_base64": audio_to_base64(audio_data, sr),
                            "sample_rate": sr,
                            "duration": duration,
                            "text": complete_text,
                        })
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "error": str(e),
                            "text": complete_text,
                        })

            elif msg_type == "flush":
                if session:
                    remaining = session.flush()
                    if remaining:
                        try:
                            audio_data, sr = generate_audio(
                                text=remaining,
                                voice_name=session.voice_name,
                                language=session.language,
                            )
                            await websocket.send_json({
                                "type": "audio",
                                "audio_base64": audio_to_base64(audio_data, sr),
                                "sample_rate": sr,
                                "duration": len(audio_data) / sr,
                                "text": remaining,
                            })
                        except Exception as e:
                            await websocket.send_json({
                                "type": "error",
                                "error": str(e),
                            })

            elif msg_type == "end":
                # Flush remaining text
                if session:
                    remaining = session.flush()
                    if remaining:
                        try:
                            audio_data, sr = generate_audio(
                                text=remaining,
                                voice_name=session.voice_name,
                                language=session.language,
                            )
                            await websocket.send_json({
                                "type": "audio",
                                "audio_base64": audio_to_base64(audio_data, sr),
                                "sample_rate": sr,
                                "duration": len(audio_data) / sr,
                                "text": remaining,
                            })
                        except Exception as e:
                            await websocket.send_json({
                                "type": "error",
                                "error": str(e),
                            })

                await websocket.send_json({"type": "done"})
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        traceback.print_exc()
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
