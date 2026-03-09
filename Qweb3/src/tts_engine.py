"""
Core TTS Engine for Qwen3-TTS Voice Cloning System
"""

import warnings
from pathlib import Path
from typing import Generator, Optional, Union

import numpy as np
import torch

from .models import VoiceProfile, GenerationResult


class TTSEngine:
    """
    Core TTS engine wrapping Qwen3-TTS for voice cloning and generation.

    The model is loaded lazily on first use to speed up startup.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        use_flash_attention: bool = True,
        quantization: Optional[str] = None,
        use_torch_compile: bool = False,
    ):
        """
        Initialize the TTS engine.

        Args:
            model_name: HuggingFace model name or local path
            device: Device to use (auto-detected if None)
            dtype: Data type ('bfloat16' or 'float16')
            use_flash_attention: Whether to use FlashAttention 2 if available
            quantization: Quantization mode ('4bit', '8bit', or None)
            use_torch_compile: Whether to use torch.compile() for speedup
        """
        self.model_name = model_name
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention
        self.quantization = quantization
        self.use_torch_compile = use_torch_compile
        self._model = None
        self._device = device

        # Auto-detect device
        if self._device is None:
            if torch.cuda.is_available():
                self._device = "cuda:0"
            else:
                self._device = "cpu"
                warnings.warn("CUDA not available, using CPU. This will be slow.")

    @property
    def device(self) -> str:
        """Get the device being used."""
        return self._device

    @property
    def model(self):
        """Lazy load the model on first access."""
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """Load the Qwen3-TTS model with optional quantization."""
        from qwen_tts import Qwen3TTSModel

        # Determine dtype
        if self.dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16

        # Check for attention implementation
        attn_impl = None
        if self.use_flash_attention:
            try:
                import flash_attn
                attn_impl = "flash_attention_2"
                print("Using flash_attention_2")
            except ImportError:
                # Fall back to SDPA (PyTorch 2.0+ built-in efficient attention)
                attn_impl = "sdpa"
                print("Using SDPA (PyTorch scaled_dot_product_attention)")

        # Build load kwargs
        load_kwargs = {
            "device_map": self._device,
            "dtype": torch_dtype,
        }
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl

        # Add quantization config
        if self.quantization:
            try:
                from transformers import BitsAndBytesConfig

                if self.quantization == "4bit":
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_dtype,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    print("Using 4-bit quantization (NF4)")
                elif self.quantization == "8bit":
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                    print("Using 8-bit quantization")
            except ImportError:
                print("WARNING: bitsandbytes not available, skipping quantization")

        print(f"Loading model: {self.model_name}")
        print(f"Device: {self._device}, dtype: {self.dtype}, quantization: {self.quantization or 'none'}")

        self._model = Qwen3TTSModel.from_pretrained(
            self.model_name,
            **load_kwargs
        )

        # Apply torch.compile for additional speedup
        if self.use_torch_compile and hasattr(torch, 'compile'):
            try:
                print("Applying torch.compile() for optimization...")
                # Compile the model for faster inference
                self._model = torch.compile(self._model, mode="reduce-overhead")
                print("torch.compile() applied successfully!")
            except Exception as e:
                print(f"WARNING: torch.compile() failed: {e}")

        print("Model loaded successfully!")

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None

    def clone_voice(
        self,
        ref_audio: Union[str, Path, np.ndarray, tuple],
        ref_text: str,
        name: str,
        language: str = "English",
    ) -> VoiceProfile:
        """
        Clone a voice from reference audio.

        Args:
            ref_audio: Reference audio (path, URL, numpy array, or (array, sr) tuple)
            ref_text: Transcription of the reference audio
            name: Name for the voice profile
            language: Language of the reference audio

        Returns:
            VoiceProfile with cached prompt items
        """
        # Convert Path to string if needed
        if isinstance(ref_audio, Path):
            ref_audio = str(ref_audio)

        # Create the voice clone prompt (this caches the voice characteristics)
        prompt_items = self.model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
        )

        return VoiceProfile(
            name=name,
            prompt_items=prompt_items,
            ref_text=ref_text,
            language=language,
        )

    def generate(
        self,
        text: str,
        voice_profile: VoiceProfile,
        language: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate speech from text using a voice profile.

        Args:
            text: Text to synthesize
            voice_profile: Voice profile to use
            language: Language override (uses profile language if None)

        Returns:
            GenerationResult with audio data
        """
        lang = language or voice_profile.language

        # Generate audio
        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language=lang,
            voice_clone_prompt=voice_profile.prompt_items,
        )

        audio_data = wavs[0]  # First (and only) result
        duration = len(audio_data) / sr

        return GenerationResult(
            audio_data=audio_data,
            sample_rate=sr,
            duration_seconds=duration,
            voice_name=voice_profile.name,
            text=text,
        )

    def generate_batch(
        self,
        texts: list[str],
        voice_profile: VoiceProfile,
        languages: Optional[list[str]] = None,
    ) -> list[GenerationResult]:
        """
        Generate speech for multiple texts using the same voice profile.

        Args:
            texts: List of texts to synthesize
            voice_profile: Voice profile to use
            languages: List of languages (uses profile language if None)

        Returns:
            List of GenerationResult objects
        """
        if languages is None:
            languages = [voice_profile.language] * len(texts)

        # Generate all at once (more efficient)
        wavs, sr = self.model.generate_voice_clone(
            text=texts,
            language=languages,
            voice_clone_prompt=voice_profile.prompt_items,
        )

        results = []
        for i, (wav, text) in enumerate(zip(wavs, texts)):
            results.append(GenerationResult(
                audio_data=wav,
                sample_rate=sr,
                duration_seconds=len(wav) / sr,
                voice_name=voice_profile.name,
                text=text,
            ))

        return results

    def generate_direct(
        self,
        text: str,
        ref_audio: Union[str, Path],
        ref_text: str,
        language: str = "English",
    ) -> GenerationResult:
        """
        Generate speech directly without creating a saved voice profile.

        Args:
            text: Text to synthesize
            ref_audio: Reference audio for voice cloning
            ref_text: Transcription of reference audio
            language: Language for generation

        Returns:
            GenerationResult with audio data
        """
        if isinstance(ref_audio, Path):
            ref_audio = str(ref_audio)

        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )

        audio_data = wavs[0]

        return GenerationResult(
            audio_data=audio_data,
            sample_rate=sr,
            duration_seconds=len(audio_data) / sr,
            voice_name="direct",
            text=text,
        )
