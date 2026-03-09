"""
TTS Client for Qweb3 Voice Cloning System

Uses WebSocket streaming for low-latency audio generation.
"""

import asyncio
import websockets
import json
import base64
import io
import wave
import numpy as np
from typing import AsyncGenerator, Optional
import time

import config


class TTSClient:
    """
    WebSocket client for Qweb3 TTS streaming.

    Supports bidirectional streaming: tokens in, audio out.
    Uses sentence-level chunking for faster response.
    """

    def __init__(self):
        self.ws_url = getattr(config, 'TTS_WS_URL', 'ws://localhost:8000/generate/realtime')
        self.http_url = getattr(config, 'TTS_HTTP_URL', 'http://localhost:8000')
        self.voice = getattr(config, 'TTS_VOICE', 'ScarletSell')
        self.language = getattr(config, 'TTS_LANGUAGE', 'English')
        self._ws = None

    async def connect(self):
        """Establish WebSocket connection."""
        try:
            self._ws = await websockets.connect(self.ws_url, ping_interval=30)

            # Send init message with voice
            init_msg = {
                "type": "init",
                "voice_name": self.voice,
                "language": self.language
            }
            await self._ws.send(json.dumps(init_msg))

            # Wait for ready
            response = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
            data = json.loads(response)
            if data.get("type") == "error":
                raise Exception(f"TTS init failed: {data.get('error')}")
            if data.get("type") != "ready":
                raise Exception(f"Unexpected response: {data}")

            if config.DEBUG:
                print(f"[TTS] Connected to {self.ws_url} with voice '{self.voice}'")

        except Exception as e:
            print(f"[TTS] Connection failed: {e}")
            self._ws = None
            raise

    async def disconnect(self):
        """Close WebSocket connection."""
        if self._ws:
            try:
                await self._ws.close()
            except:
                pass
            self._ws = None

    async def stream_generate(
        self,
        token_generator: AsyncGenerator[str, None]
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Stream tokens to TTS and yield audio chunks as they arrive.

        Qweb3 generates audio on sentence boundaries for faster response.

        Args:
            token_generator: Async generator yielding text tokens

        Yields:
            Audio chunks as numpy arrays (int16)
        """
        # Always reconnect for each generation (server closes after done)
        if self._ws is not None:
            try:
                await self._ws.close()
            except:
                pass
            self._ws = None
        await self.connect()

        start = time.perf_counter()
        first_audio_time = None

        # Queue for received audio
        audio_queue = asyncio.Queue()
        send_done = asyncio.Event()

        async def send_tokens():
            """Send tokens to the WebSocket."""
            try:
                async for token in token_generator:
                    if self._ws:
                        msg = {"type": "token", "token": token}
                        await self._ws.send(json.dumps(msg))
                # Signal end of tokens
                if self._ws:
                    await self._ws.send(json.dumps({"type": "end"}))
            except Exception as e:
                print(f"[TTS] Send error: {e}")
            finally:
                send_done.set()

        async def receive_audio():
            """Receive audio from the WebSocket and put in queue."""
            nonlocal first_audio_time
            try:
                while True:
                    response = await self._ws.recv()
                    data = json.loads(response)

                    if data.get("type") == "audio":
                        if first_audio_time is None:
                            first_audio_time = time.perf_counter()
                            if config.LOG_LATENCY:
                                ttfa = (first_audio_time - start) * 1000
                                print(f"[TTS] First audio in {ttfa:.0f}ms")

                        # Decode base64 WAV audio
                        audio_bytes = base64.b64decode(data["audio_base64"])
                        # Parse WAV to get raw PCM
                        wav_io = io.BytesIO(audio_bytes)
                        try:
                            with wave.open(wav_io, 'rb') as wav_file:
                                audio = np.frombuffer(
                                    wav_file.readframes(-1),
                                    dtype=np.int16
                                )
                            await audio_queue.put(audio)
                        except Exception as e:
                            # Maybe raw PCM, try direct decode
                            audio = np.frombuffer(audio_bytes, dtype=np.int16)
                            await audio_queue.put(audio)

                    elif data.get("type") == "done":
                        break

                    elif data.get("type") == "error":
                        print(f"[TTS] Error: {data.get('error')}")
                        break
            except websockets.exceptions.ConnectionClosed:
                pass
            except Exception as e:
                print(f"[TTS] Receive error: {e}")
            finally:
                await audio_queue.put(None)  # Signal end

        # Start both tasks
        send_task = asyncio.create_task(send_tokens())
        receive_task = asyncio.create_task(receive_audio())

        # Yield audio chunks as they arrive
        try:
            while True:
                chunk = await audio_queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            # Wait for tasks to complete
            await send_task
            await receive_task

            # Log total time
            if first_audio_time and config.LOG_LATENCY:
                total = (time.perf_counter() - start) * 1000
                print(f"[TTS] Total generation: {total:.0f}ms")

    async def health_check(self) -> bool:
        """Check if the TTS server is available."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.http_url}/", timeout=5.0)
                return response.status_code == 200
        except Exception:
            return False

    async def list_voices(self) -> list:
        """List available voices from the server."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.http_url}/voices", timeout=5.0)
                if response.status_code == 200:
                    data = response.json()
                    return [v["name"] for v in data.get("voices", [])]
        except Exception as e:
            print(f"[TTS] Failed to list voices: {e}")
        return []
