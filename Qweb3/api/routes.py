"""
FastAPI route handlers for TTS API
"""

import asyncio
import base64
import json
import re
import tempfile
import traceback
from pathlib import Path
from typing import Optional
from collections import deque

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

import config
from src import (
    TTSEngine,
    VoiceManager,
    load_audio,
    validate_reference_audio,
    save_audio,
    AudioStreamer,
    stream_audio_sse,
    analyze_audio,
    compare_audio,
)

from .models import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    LanguageResponse,
    MessageResponse,
    VoiceListResponse,
    VoiceResponse,
    ErrorResponse,
    AnalysisResponse,
    SpectralMetricsResponse,
    ProsodyMetricsResponse,
    QualityMetricsResponse,
    ComparisonResponse,
)
from .dependencies import get_engine, get_voice_manager


router = APIRouter()


# Health Check

@router.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse()


# Voice Management

@router.get("/voices", response_model=VoiceListResponse, tags=["Voices"])
async def list_voices(
    manager: VoiceManager = Depends(get_voice_manager),
):
    """List all saved voice profiles."""
    voices = manager.list_voices()
    return VoiceListResponse(
        voices=[
            VoiceResponse(
                name=v.name,
                ref_text=v.ref_text,
                language=v.language,
                created_at=v.created_at,
            )
            for v in voices
        ],
        count=len(voices),
    )


@router.get("/voices/{name}", response_model=VoiceResponse, tags=["Voices"])
async def get_voice(
    name: str,
    manager: VoiceManager = Depends(get_voice_manager),
):
    """Get information about a specific voice profile."""
    voice = manager.load_voice(name)
    if voice is None:
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")

    return VoiceResponse(
        name=voice.name,
        ref_text=voice.ref_text,
        language=voice.language,
        created_at=voice.created_at,
    )


@router.post("/voices/clone", response_model=VoiceResponse, tags=["Voices"])
async def clone_voice(
    name: str = Form(..., description="Name for the voice profile"),
    ref_text: str = Form(..., description="Transcription of the reference audio"),
    language: str = Form("English", description="Language of the reference"),
    audio: UploadFile = File(..., description="Reference audio file"),
    engine: TTSEngine = Depends(get_engine),
    manager: VoiceManager = Depends(get_voice_manager),
):
    """
    Clone a voice from uploaded audio.

    Upload a 3-30 second audio file of the voice you want to clone,
    along with a transcription of what is being said in the audio.
    """
    # Check if voice already exists
    if manager.voice_exists(name):
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{name}' already exists. Delete it first or use a different name."
        )

    # Validate file type
    if audio.content_type and not audio.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {audio.content_type}. Must be an audio file."
        )

    # Save uploaded file temporarily
    temp_path = None
    try:
        # Get file extension from filename
        suffix = Path(audio.filename).suffix if audio.filename else ".wav"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio.read()
            tmp.write(content)
            temp_path = tmp.name

        # Load and validate audio
        try:
            audio_data, sr = load_audio(temp_path)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load audio: {str(e)}"
            )

        is_valid, error = validate_reference_audio(audio_data, sr)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error)

        # Clone the voice
        try:
            voice_profile = engine.clone_voice(
                ref_audio=temp_path,
                ref_text=ref_text,
                name=name,
                language=language,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Voice cloning failed: {str(e)}"
            )

        # Save the voice profile
        manager.save_voice(voice_profile)

        return VoiceResponse(
            name=voice_profile.name,
            ref_text=voice_profile.ref_text,
            language=voice_profile.language,
            created_at=voice_profile.created_at,
        )

    finally:
        # Clean up temp file
        if temp_path:
            try:
                Path(temp_path).unlink()
            except:
                pass


@router.delete("/voices/{name}", response_model=MessageResponse, tags=["Voices"])
async def delete_voice(
    name: str,
    manager: VoiceManager = Depends(get_voice_manager),
):
    """Delete a voice profile."""
    if not manager.voice_exists(name):
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")

    manager.delete_voice(name)
    return MessageResponse(message=f"Voice '{name}' deleted successfully")


# TTS Generation

@router.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate_speech(
    request: GenerateRequest,
    engine: TTSEngine = Depends(get_engine),
    manager: VoiceManager = Depends(get_voice_manager),
):
    """
    Generate speech from text using a saved voice profile.

    Returns base64-encoded WAV audio.
    """
    # Load voice profile
    voice = manager.load_voice(request.voice_name)
    if voice is None:
        raise HTTPException(
            status_code=404,
            detail=f"Voice '{request.voice_name}' not found"
        )

    try:
        # Generate speech
        result = engine.generate(
            text=request.text,
            voice_profile=voice,
            language=request.language,
        )

        # Convert to base64
        streamer = AudioStreamer(sample_rate=result.sample_rate)
        audio_b64 = streamer.audio_to_base64(result.audio_data)

        return GenerateResponse(
            audio_base64=audio_b64,
            sample_rate=result.sample_rate,
            duration_seconds=result.duration_seconds,
            text=request.text,
            voice_name=request.voice_name,
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )


@router.post("/generate/stream", tags=["Generation"])
async def generate_speech_stream(
    request: GenerateRequest,
    engine: TTSEngine = Depends(get_engine),
    manager: VoiceManager = Depends(get_voice_manager),
):
    """
    Generate speech with streaming response.

    Returns Server-Sent Events with base64-encoded audio chunks.
    """
    # Load voice profile
    voice = manager.load_voice(request.voice_name)
    if voice is None:
        raise HTTPException(
            status_code=404,
            detail=f"Voice '{request.voice_name}' not found"
        )

    async def generate_sse():
        try:
            # Generate speech
            result = engine.generate(
                text=request.text,
                voice_profile=voice,
                language=request.language,
            )

            # Stream as SSE
            for event in stream_audio_sse(
                result.audio_data,
                result.sample_rate,
                chunk_duration=0.5,
            ):
                yield event

        except Exception as e:
            error_event = f'event: error\ndata: {{"error": "{str(e)}"}}\n\n'
            yield error_event

    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# Languages

@router.get("/languages", response_model=LanguageResponse, tags=["Languages"])
async def list_languages():
    """List supported languages for TTS."""
    return LanguageResponse(
        languages=config.SUPPORTED_LANGUAGES,
        default=config.DEFAULT_LANGUAGE,
    )


# Audio Analysis

@router.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_audio_endpoint(
    audio: UploadFile = File(..., description="Audio file to analyze"),
):
    """
    Analyze an audio file and return quality metrics.

    Returns spectral, prosody, and quality metrics for the uploaded audio.
    """
    # Validate file type (allow octet-stream for curl uploads)
    allowed_types = ["audio/", "application/octet-stream"]
    if audio.content_type and not any(audio.content_type.startswith(t) for t in allowed_types):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {audio.content_type}. Must be an audio file."
        )

    # Save uploaded file temporarily
    temp_path = None
    try:
        suffix = Path(audio.filename).suffix if audio.filename else ".wav"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio.read()
            tmp.write(content)
            temp_path = tmp.name

        # Load audio
        try:
            audio_data, sr = load_audio(temp_path)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load audio: {str(e)}"
            )

        # Analyze audio
        try:
            analysis = analyze_audio(audio_data, sr)

            return AnalysisResponse(
                duration_seconds=analysis.duration_seconds,
                sample_rate=analysis.sample_rate,
                num_samples=analysis.num_samples,
                spectral=SpectralMetricsResponse(**analysis.spectral.to_dict()),
                prosody=ProsodyMetricsResponse(**analysis.prosody.to_dict()),
                quality=QualityMetricsResponse(**analysis.quality.to_dict()),
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {str(e)}"
            )

    finally:
        # Clean up temp file
        if temp_path:
            try:
                Path(temp_path).unlink()
            except:
                pass


@router.post("/compare", response_model=ComparisonResponse, tags=["Analysis"])
async def compare_audio_endpoint(
    audio1: UploadFile = File(..., description="First audio file"),
    audio2: UploadFile = File(..., description="Second audio file"),
):
    """
    Compare two audio files and return similarity metrics.

    Useful for comparing generated audio against reference audio
    or A/B testing different TTS outputs.
    """
    temp_path1 = None
    temp_path2 = None

    try:
        # Save first file
        suffix1 = Path(audio1.filename).suffix if audio1.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix1) as tmp:
            content = await audio1.read()
            tmp.write(content)
            temp_path1 = tmp.name

        # Save second file
        suffix2 = Path(audio2.filename).suffix if audio2.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix2) as tmp:
            content = await audio2.read()
            tmp.write(content)
            temp_path2 = tmp.name

        # Load both audio files
        try:
            audio_data1, sr1 = load_audio(temp_path1)
            audio_data2, sr2 = load_audio(temp_path2)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load audio: {str(e)}"
            )

        # Compare audio
        try:
            comparison = compare_audio(audio_data1, sr1, audio_data2, sr2)

            return ComparisonResponse(
                duration_diff_seconds=comparison["duration_diff_seconds"],
                centroid_diff=comparison["centroid_diff"],
                bandwidth_diff=comparison["bandwidth_diff"],
                pitch_mean_diff=comparison["pitch_mean_diff"],
                pitch_std_diff=comparison["pitch_std_diff"],
                rms_diff=comparison["rms_diff"],
                dynamic_range_diff=comparison["dynamic_range_diff"],
                overall_similarity=comparison["overall_similarity"],
            )
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Comparison failed: {str(e)}"
            )

    finally:
        # Clean up temp files
        for path in [temp_path1, temp_path2]:
            if path:
                try:
                    Path(path).unlink()
                except:
                    pass


# Realtime Bidirectional Streaming

class RealtimeTTSSession:
    """Manages a realtime TTS streaming session."""

    def __init__(self, voice_profile, language: str, engine: TTSEngine):
        self.voice_profile = voice_profile
        self.language = language
        self.engine = engine
        self.text_buffer = ""
        self.audio_queue = asyncio.Queue()
        self.is_running = True
        # Sentence-ending punctuation
        self.sentence_endings = re.compile(r'[.!?;]\s*')
        # Clause-ending punctuation (for faster response)
        self.clause_endings = re.compile(r'[,:\-]\s*')
        self.min_chunk_length = 10  # Minimum chars before generating on clause

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

    def generate_audio_sync(self, text: str) -> dict:
        """Generate audio for text and return as base64 (synchronous)."""
        try:
            result = self.engine.generate(
                text=text,
                voice_profile=self.voice_profile,
                language=self.language,
            )

            # Convert to base64
            streamer = AudioStreamer(sample_rate=result.sample_rate)
            audio_b64 = streamer.audio_to_base64(result.audio_data)

            return {
                "type": "audio",
                "audio_base64": audio_b64,
                "sample_rate": result.sample_rate,
                "duration": result.duration_seconds,
                "text": text,
            }
        except Exception as e:
            traceback.print_exc()
            return {
                "type": "error",
                "error": str(e),
                "text": text,
            }


@router.websocket("/generate/realtime")
async def realtime_tts_stream(websocket: WebSocket):
    """
    Bidirectional WebSocket for realtime TTS streaming.

    Send JSON messages:
    - {"type": "init", "voice_name": "...", "language": "English"}
    - {"type": "token", "token": "Hello"}
    - {"type": "token", "token": " world"}
    - {"type": "flush"}  (generate remaining buffer)
    - {"type": "end"}

    Receive JSON messages:
    - {"type": "ready"}
    - {"type": "audio", "audio_base64": "...", "sample_rate": 24000, "duration": 1.5, "text": "..."}
    - {"type": "error", "error": "..."}
    - {"type": "done"}
    """
    await websocket.accept()

    # Get engine and manager directly (Depends doesn't work with WebSockets)
    engine = get_engine()
    manager = get_voice_manager()

    session = None

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            msg = json.loads(data)
            msg_type = msg.get("type", "")

            if msg_type == "init":
                # Initialize session with voice
                voice_name = msg.get("voice_name")
                language = msg.get("language", "English")

                voice = manager.load_voice(voice_name)
                if voice is None:
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Voice '{voice_name}' not found"
                    })
                    continue

                session = RealtimeTTSSession(voice, language, engine)
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
                    # Generate audio and send back
                    result = session.generate_audio_sync(complete_text)
                    await websocket.send_json(result)

            elif msg_type == "flush":
                if session:
                    remaining = session.flush()
                    if remaining:
                        result = session.generate_audio_sync(remaining)
                        await websocket.send_json(result)

            elif msg_type == "end":
                # Flush any remaining text
                if session:
                    remaining = session.flush()
                    if remaining:
                        result = session.generate_audio_sync(remaining)
                        await websocket.send_json(result)

                await websocket.send_json({"type": "done"})
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        traceback.print_exc()
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
        except:
            pass
