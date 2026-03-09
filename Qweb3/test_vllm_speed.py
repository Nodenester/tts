"""Test vLLM TTS speed."""
import requests
import time
import base64
import io

url = "http://localhost:8000/generate"
test_texts = [
    "Hello world, this is a quick test.",
    "The quick brown fox jumps over the lazy dog. This is a longer sentence to test the performance.",
    "Testing text to speech generation with the vLLM Omni backend. We want to see how fast this can generate audio compared to the old slow implementation."
]

print("Testing vLLM-Omni TTS Speed")
print("=" * 50)

for i, text in enumerate(test_texts):
    print(f"\nTest {i+1}: {len(text)} chars")
    print(f"Text: {text[:50]}...")

    start = time.perf_counter()
    response = requests.post(url, json={
        "text": text,
        "language": "English"
    })
    elapsed = time.perf_counter() - start

    if response.status_code == 200:
        data = response.json()
        duration = data["duration"]
        rtf = elapsed / duration  # Real-time factor

        print(f"  Generation time: {elapsed:.3f}s")
        print(f"  Audio duration: {duration:.3f}s")
        print(f"  Real-time factor: {rtf:.2f}x")
        print(f"  Speed: {duration/elapsed:.1f}x faster than real-time")
    else:
        print(f"  Error: {response.status_code}")
        print(f"  {response.text}")
