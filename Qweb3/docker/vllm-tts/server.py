"""
Fast Qwen3-TTS server using vLLM-Omni.
Based on: https://github.com/vllm-project/vllm-omni
"""
import asyncio
import base64
import io
import json
import os
import tempfile
from typing import Optional, Dict, Any, List

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

# vLLM-Omni imports
from vllm_omni.entrypoints.omni import Omni
from vllm.sampling_params import SamplingParams

app = FastAPI(title="Qwen3-TTS vLLM Server")

# Global model instance
omni: Optional[Omni] = None
voice_profiles: Dict[str, Any] = {}

MODEL_NAME = os.getenv("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
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
    """Load the TTS model with vLLM-Omni."""
    global omni
    print(f"Loading model: {MODEL_NAME}")

    omni = Omni(
        model=MODEL_NAME,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        enforce_eager=False,  # Enable CUDA graphs for speed
    )
    print("Model loaded!")
    return omni


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


def build_tts_prompt(text: str, language: str = "English") -> str:
    """Build the TTS prompt in Qwen3-TTS format."""
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def build_voice_clone_inputs(
    text: str,
    ref_audio_path: str,
    ref_text: str,
    language: str = "English",
) -> Dict[str, Any]:
    """Build inputs for voice cloning (Base model)."""
    prompt = build_tts_prompt(text, language)

    return {
        "prompt": prompt,
        "additional_information": {
            "task_type": ["Base"],
            "text": [text],
            "language": [language],
            "ref_audio": [ref_audio_path],
            "ref_text": [ref_text],
            "mode_tag": ["icl"],  # In-context learning mode for voice cloning
            "max_new_tokens": [2048],
        }
    }


@app.on_event("startup")
async def startup():
    """Load model on startup."""
    load_model()


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "loaded": omni is not None}


@app.get("/voices")
async def list_voices():
    """List available voice profiles."""
    return {"voices": list(voice_profiles.keys())}


@app.post("/clone_voice")
async def clone_voice(request: VoiceCloneRequest):
    """Create a voice profile from reference audio."""
    try:
        # Decode and save reference audio to temp file
        ref_audio, ref_sr = base64_to_audio(request.ref_audio_base64)

        # Save to temp file (vLLM-Omni needs file path)
        temp_dir = tempfile.mkdtemp()
        ref_audio_path = os.path.join(temp_dir, f"{request.name}_ref.wav")
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
    if omni is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        voice_profile = voice_profiles.get(request.voice_name) if request.voice_name else None

        if voice_profile:
            # Voice cloning mode
            inputs = build_voice_clone_inputs(
                text=request.text,
                ref_audio_path=voice_profile["ref_audio_path"],
                ref_text=voice_profile["ref_text"],
                language=request.language,
            )
        else:
            # Simple TTS (no voice cloning)
            inputs = {
                "prompt": build_tts_prompt(request.text, request.language),
                "additional_information": {
                    "task_type": ["Base"],
                    "text": [request.text],
                    "language": [request.language],
                    "max_new_tokens": [2048],
                }
            }

        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=0.9,
            top_p=1.0,
            top_k=50,
            max_tokens=2048,
        )

        # Generate with vLLM-Omni
        generator = omni.generate([inputs], [sampling_params])

        # Extract audio from output
        audio_data = None
        sample_rate = SAMPLE_RATE

        for stage_outputs in generator:
            for output in stage_outputs.request_output:
                if hasattr(output, 'multimodal_output') and output.multimodal_output:
                    audio_tensor = output.multimodal_output.get("audio")
                    if audio_tensor is not None:
                        audio_data = audio_tensor.float().detach().cpu().numpy()
                    sr = output.multimodal_output.get("sr")
                    if sr is not None:
                        sample_rate = sr.item() if hasattr(sr, 'item') else int(sr)

        if audio_data is None:
            raise HTTPException(status_code=500, detail="No audio generated")

        audio_b64 = audio_to_base64(audio_data, sample_rate)
        duration = len(audio_data) / sample_rate

        return GenerateResponse(
            audio_base64=audio_b64,
            sample_rate=sample_rate,
            duration=duration,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for streaming TTS."""
    await websocket.accept()

    voice_name = None
    tokens_buffer: List[str] = []

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "config":
                voice_name = msg.get("voice_name")
                await websocket.send_json({"type": "ready"})

            elif msg.get("type") == "token":
                token = msg.get("token", "")
                tokens_buffer.append(token)

            elif msg.get("type") == "end":
                # Generate audio for accumulated text
                full_text = "".join(tokens_buffer)

                if full_text.strip() and omni is not None:
                    voice_profile = voice_profiles.get(voice_name) if voice_name else None

                    if voice_profile:
                        inputs = build_voice_clone_inputs(
                            text=full_text,
                            ref_audio_path=voice_profile["ref_audio_path"],
                            ref_text=voice_profile["ref_text"],
                            language=voice_profile.get("language", "English"),
                        )
                    else:
                        inputs = {
                            "prompt": build_tts_prompt(full_text),
                            "additional_information": {
                                "task_type": ["Base"],
                                "text": [full_text],
                                "language": ["English"],
                                "max_new_tokens": [2048],
                            }
                        }

                    sampling_params = SamplingParams(
                        temperature=0.9,
                        top_p=1.0,
                        top_k=50,
                        max_tokens=2048,
                    )

                    generator = omni.generate([inputs], [sampling_params])

                    for stage_outputs in generator:
                        for output in stage_outputs.request_output:
                            if hasattr(output, 'multimodal_output') and output.multimodal_output:
                                audio_tensor = output.multimodal_output.get("audio")
                                if audio_tensor is not None:
                                    audio_data = audio_tensor.float().detach().cpu().numpy()
                                    sr = output.multimodal_output.get("sr")
                                    sample_rate = sr.item() if sr is not None and hasattr(sr, 'item') else SAMPLE_RATE

                                    audio_b64 = audio_to_base64(audio_data, sample_rate)
                                    await websocket.send_json({
                                        "type": "audio",
                                        "audio_base64": audio_b64,
                                        "sample_rate": sample_rate,
                                    })

                await websocket.send_json({"type": "done"})
                tokens_buffer = []

    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
