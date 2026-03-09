"""
Fast F5-TTS server with voice cloning support.
"""
import base64
import io
import os
import tempfile
import time
from typing import Optional, Dict, Any

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="F5-TTS Server")

# Global model instance
f5_model = None
voice_profiles: Dict[str, Any] = {}

SAMPLE_RATE = 24000


class TTSRequest(BaseModel):
    text: str
    voice_name: Optional[str] = None
    language: str = "English"


class VoiceCloneRequest(BaseModel):
    name: str
    ref_audio_base64: str
    ref_text: str
    language: str = "English"


class GenerateResponse(BaseModel):
    audio_base64: str
    sample_rate: int
    duration: float


def load_model():
    """Load the F5-TTS model."""
    global f5_model
    print("Loading F5-TTS model...")

    from f5_tts.api import F5TTS

    f5_model = F5TTS(
        model="F5TTS_v1_Base",
        ckpt_file="",
        vocab_file="",
    )
    print("F5-TTS model loaded!")
    return f5_model


def audio_to_base64(audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Convert audio numpy array to base64 WAV."""
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def base64_to_audio(b64_string: str) -> tuple[np.ndarray, int]:
    """Convert base64 WAV to numpy array."""
    audio_bytes = base64.b64decode(b64_string)
    buffer = io.BytesIO(audio_bytes)
    audio_data, sr = sf.read(buffer)
    return audio_data.astype(np.float32), sr


@app.on_event("startup")
async def startup():
    """Load model on startup."""
    load_model()


@app.get("/health")
async def health():
    return {"status": "ok", "model": "F5-TTS", "loaded": f5_model is not None}


@app.get("/voices")
async def list_voices():
    """List available voice profiles."""
    return {"voices": list(voice_profiles.keys())}


@app.post("/clone_voice")
async def clone_voice(request: VoiceCloneRequest):
    """Create a voice profile from reference audio."""
    try:
        ref_audio, ref_sr = base64_to_audio(request.ref_audio_base64)

        # Ensure voices directory exists
        os.makedirs("/app/voices", exist_ok=True)

        # Save reference audio
        ref_audio_path = f"/app/voices/{request.name}_ref.wav"
        sf.write(ref_audio_path, ref_audio, ref_sr)

        # Store voice profile
        voice_profiles[request.name] = {
            "ref_audio_path": ref_audio_path,
            "ref_text": request.ref_text,
            "language": request.language,
        }

        return {"status": "ok", "voice_name": request.name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: TTSRequest):
    """Generate speech from text."""
    if f5_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        start_time = time.perf_counter()

        voice_profile = voice_profiles.get(request.voice_name) if request.voice_name else None

        if voice_profile:
            ref_audio_path = voice_profile["ref_audio_path"]
            ref_text = voice_profile["ref_text"]
        else:
            # Use a default reference if no voice specified
            # F5-TTS requires a reference audio for voice cloning
            raise HTTPException(
                status_code=400,
                detail="F5-TTS requires a voice profile. Create one first with /clone_voice"
            )

        # Generate using F5-TTS
        audio_data, sample_rate, _ = f5_model.infer(
            ref_file=ref_audio_path,
            ref_text=ref_text,
            gen_text=request.text,
            show_info=print,
        )

        elapsed = time.perf_counter() - start_time

        # Convert to numpy if tensor
        if torch.is_tensor(audio_data):
            audio_data = audio_data.float().detach().cpu().numpy()

        audio_b64 = audio_to_base64(audio_data, sample_rate)
        duration = len(audio_data) / sample_rate

        print(f"Generated {duration:.2f}s audio in {elapsed:.2f}s (RTF: {elapsed/duration:.2f})")

        return GenerateResponse(
            audio_base64=audio_b64,
            sample_rate=sample_rate,
            duration=duration,
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
