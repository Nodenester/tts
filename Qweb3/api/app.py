"""
FastAPI application factory
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router
from .dependencies import reset_singletons


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("Starting TTS API server...")
    yield
    # Shutdown
    print("Shutting down TTS API server...")
    reset_singletons()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Qwen3-TTS Voice Cloning API",
        description="""
# Qwen3-TTS Voice Cloning System

A text-to-speech API with voice cloning capabilities powered by Qwen3-TTS.

## Features

- **Voice Cloning**: Clone any voice from a 3-30 second audio sample
- **Text-to-Speech**: Generate natural speech from text
- **Streaming**: Real-time audio streaming support
- **Multi-language**: Support for 10 languages

## Quick Start

1. **Clone a voice**: Upload reference audio with transcription
2. **Generate speech**: Use the cloned voice to synthesize text
3. **Stream audio**: Get real-time audio chunks for playback

## Ethical Use

This API is designed for legitimate use cases only:
- Clone your own voice
- Clone voices with explicit consent
- Research and evaluation purposes

Do not use for impersonation or creating misleading content.
        """,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for development
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(router)

    return app


# Create the app instance
app = create_app()
