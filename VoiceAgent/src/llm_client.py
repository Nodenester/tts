"""
LLM Client for llama.cpp server

Streams responses token-by-token for minimal latency.
"""

import httpx
import json
from typing import AsyncGenerator
import time

import config


class LLMClient:
    """
    Client for llama.cpp server with streaming support.

    Yields tokens as they are generated for minimum latency.
    """

    def __init__(self):
        self.base_url = config.LLM_URL
        self.system_prompt = config.LLM_SYSTEM_PROMPT
        self.conversation_history = []

    def reset(self):
        """Reset conversation history."""
        self.conversation_history = []

    async def generate_stream(self, user_message: str) -> AsyncGenerator[str, None]:
        """
        Generate a response, yielding tokens as they arrive.

        Args:
            user_message: The user's input text

        Yields:
            Tokens as they are generated
        """
        start = time.perf_counter()
        first_token_time = None

        # Build messages
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_message})

        # Request payload
        payload = {
            "messages": messages,
            "max_tokens": config.LLM_MAX_TOKENS,
            "temperature": config.LLM_TEMPERATURE,
            "top_p": config.LLM_TOP_P,
            "stream": True
        }

        full_response = ""

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=30.0
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")

                                if content:
                                    if first_token_time is None:
                                        first_token_time = time.perf_counter()
                                        if config.LOG_LATENCY:
                                            ttft = (first_token_time - start) * 1000
                                            print(f"[LLM] First token in {ttft:.0f}ms")

                                    full_response += content
                                    yield content
                        except json.JSONDecodeError:
                            continue

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": full_response})

        # Keep history manageable
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

    async def health_check(self) -> bool:
        """Check if the LLM server is available."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health", timeout=5.0)
                return response.status_code == 200
        except Exception:
            return False
