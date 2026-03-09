"""
Example: Realtime TTS WebSocket Client

This shows how to stream text tokens to the TTS API
and receive audio chunks back in real-time.

Perfect for integrating with LLM streaming responses!
"""

import asyncio
import json
import base64
import wave
import io
from websockets import connect


async def stream_tts_realtime(voice_name: str, text_tokens: list[str]):
    """
    Stream text tokens to TTS and receive audio back.

    Args:
        voice_name: Name of the cloned voice to use
        text_tokens: List of text tokens (simulating LLM stream)
    """
    uri = "ws://localhost:8000/generate/realtime"

    all_audio = []

    async with connect(uri) as websocket:
        # 1. Initialize session with voice
        await websocket.send(json.dumps({
            "type": "init",
            "voice_name": voice_name,
            "language": "English"
        }))

        response = await websocket.recv()
        print(f"Server: {response}")

        # 2. Stream tokens (simulating LLM output)
        async def send_tokens():
            for token in text_tokens:
                print(f"Sending token: {repr(token)}")
                await websocket.send(json.dumps({
                    "type": "token",
                    "token": token
                }))
                # Simulate LLM delay between tokens
                await asyncio.sleep(0.05)

            # Flush remaining buffer and end
            await websocket.send(json.dumps({"type": "flush"}))
            await websocket.send(json.dumps({"type": "end"}))

        # 3. Receive audio chunks
        async def receive_audio():
            while True:
                response = await websocket.recv()
                msg = json.loads(response)

                if msg["type"] == "audio":
                    print(f"Received audio: {msg['duration']:.2f}s for '{msg['text']}'")
                    # Decode base64 audio
                    audio_bytes = base64.b64decode(msg["audio_base64"])
                    all_audio.append(audio_bytes)

                elif msg["type"] == "done":
                    print("Stream complete!")
                    break

                elif msg["type"] == "error":
                    print(f"Error: {msg['error']}")
                    break

        # Run both concurrently
        await asyncio.gather(send_tokens(), receive_audio())

    return all_audio


def save_audio_chunks(audio_chunks: list[bytes], output_path: str):
    """Combine and save audio chunks to a WAV file."""
    if not audio_chunks:
        print("No audio to save")
        return

    # Each chunk is already a complete WAV file,
    # we need to extract PCM data and combine
    combined_data = b""
    sample_rate = None

    for chunk in audio_chunks:
        with io.BytesIO(chunk) as f:
            with wave.open(f, 'rb') as wav:
                if sample_rate is None:
                    sample_rate = wav.getframerate()
                combined_data += wav.readframes(wav.getnframes())

    # Write combined audio
    with wave.open(output_path, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(combined_data)

    print(f"Saved combined audio to {output_path}")


async def main():
    # Simulate LLM streaming tokens
    text = "Hello! This is a test of the realtime streaming TTS system. It can handle text as it arrives, token by token."

    # Split into tokens (simulating LLM output)
    tokens = []
    for word in text.split(" "):
        tokens.append(word)
        tokens.append(" ")
    tokens = tokens[:-1]  # Remove trailing space

    print("=" * 50)
    print("Realtime TTS Streaming Demo")
    print("=" * 50)
    print(f"Voice: ScarletSell")
    print(f"Text: {text}")
    print("=" * 50)

    # Stream tokens and receive audio
    audio_chunks = await stream_tts_realtime("ScarletSell", tokens)

    # Save the result
    if audio_chunks:
        save_audio_chunks(audio_chunks, "output/realtime_output.wav")


if __name__ == "__main__":
    asyncio.run(main())
