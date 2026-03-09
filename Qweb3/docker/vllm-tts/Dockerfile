# vLLM-Omni based Qwen3-TTS server
FROM vllm/vllm-omni:v0.14.0rc1

WORKDIR /app

# Install additional dependencies
RUN pip install fastapi uvicorn soundfile numpy websockets

# Copy server code
COPY server.py /app/server.py

# Pre-download model (optional - can also mount cache)
# RUN python -c "from vllm import LLM; LLM('Qwen/Qwen3-TTS-12Hz-0.6B-Base')"

EXPOSE 8000

CMD ["python", "server.py"]
