# Qwen3-TTS with Flash Attention and PyTorch Nightly (for sm_120 support)
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    git \
    sox \
    libsox-dev \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install PyTorch nightly with CUDA 12.4 support (includes sm_120)
RUN pip install --no-cache-dir --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124

# Install numpy first (required by qwen-tts deps)
RUN pip install --no-cache-dir numpy

# Install other dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    soundfile \
    websockets \
    pydantic \
    transformers \
    accelerate

# Install qwen-tts
RUN pip install --no-cache-dir qwen-tts

# Install flash-attn (may take a while)
RUN pip install flash-attn --no-build-isolation || echo "Flash-attn build failed, will use SDPA"

WORKDIR /app

# Copy server script
COPY server.py /app/server.py

EXPOSE 8000

CMD ["python", "/app/server.py"]
