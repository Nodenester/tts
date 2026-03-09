"""
LLM Client Test

Tests llama.cpp streaming responses.
Run with: python -m tests.test_llm
"""

import asyncio
import sys
import time
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.llm_client import LLMClient


class LLMTest:
    """Test harness for LLM client."""

    def __init__(self):
        self.llm = LLMClient()

    async def run_test(self):
        """Run LLM tests."""
        print("=" * 60)
        print("LLM Client Test (llama.cpp)")
        print("=" * 60)

        # Check health
        print(f"\n[1/3] Checking LLM server at {config.LLM_URL}...")

        is_healthy = await self.llm.health_check()
        if not is_healthy:
            print("  [FAIL] LLM server not responding!")
            print("\n  To start the LLM server:")
            print("  1. Download a GGUF model (e.g., from Hugging Face)")
            print("  2. Place it in models/model.gguf")
            print("  3. Run: cd docker && docker-compose up -d")
            print("  4. Wait for the server to start")
            print("\n  Alternatively, set LLM_URL in config.py to point to")
            print("  an existing llama.cpp or compatible server.")
            return

        print("  [OK] LLM server is responding")

        # Test prompts
        test_prompts = [
            "Hello!",
            "What is 2 + 2?",
            "Tell me a short joke.",
        ]

        print(f"\n[2/3] Testing streaming responses...")
        results = []

        for i, prompt in enumerate(test_prompts):
            print(f"\n  Test {i+1}/{len(test_prompts)}: '{prompt}'")

            start = time.perf_counter()
            first_token_time = None
            tokens = []

            try:
                async for token in self.llm.generate_stream(prompt):
                    if first_token_time is None:
                        first_token_time = time.perf_counter()

                    tokens.append(token)
                    print(token, end="", flush=True)

                end_time = time.perf_counter()
                print()

                ttft = (first_token_time - start) * 1000 if first_token_time else 0
                total_time = (end_time - start) * 1000
                response = "".join(tokens)

                results.append({
                    "prompt": prompt,
                    "response": response,
                    "ttft_ms": ttft,
                    "total_ms": total_time,
                    "tokens": len(tokens)
                })

                print(f"    TTFT: {ttft:.0f}ms | Total: {total_time:.0f}ms | Tokens: {len(tokens)}")

            except Exception as e:
                print(f"\n    ERROR: {e}")
                results.append({
                    "prompt": prompt,
                    "error": str(e)
                })

        # Print summary
        self._print_summary(results)

    def _print_summary(self, results):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)

        successful = [r for r in results if "error" not in r]

        if successful:
            avg_ttft = sum(r["ttft_ms"] for r in successful) / len(successful)
            avg_total = sum(r["total_ms"] for r in successful) / len(successful)
            total_tokens = sum(r["tokens"] for r in successful)

            print(f"\nResults ({len(successful)}/{len(results)} successful):")
            print(f"  Average TTFT: {avg_ttft:.0f}ms")
            print(f"  Average total: {avg_total:.0f}ms")
            print(f"  Total tokens: {total_tokens}")

            if avg_ttft < 150:
                print(f"\n  [PASS] TTFT target (<150ms) achieved!")
            else:
                print(f"\n  [WARN] TTFT target (<150ms) not met")
        else:
            print("\n  All tests failed!")

        print("=" * 60)


async def main():
    """Run LLM test."""
    test = LLMTest()
    await test.run_test()


if __name__ == "__main__":
    asyncio.run(main())
