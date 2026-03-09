"""
Detailed profiling of TTS generation to find the bottleneck.
"""
import time
import torch
from pathlib import Path

# Force CUDA to be synchronous for accurate timing
torch.cuda.set_sync_debug_mode(1)

def profile_tts():
    from src.tts_engine import TTSEngine
    import config

    print("=" * 60)
    print("DETAILED TTS PROFILING")
    print("=" * 60)

    # Initialize engine
    print("\nInitializing TTS engine...")
    engine = TTSEngine(
        model_name=config.MODEL_NAME,
        dtype=config.MODEL_DTYPE,
        use_flash_attention=config.USE_FLASH_ATTENTION,
    )

    # Force model load
    model = engine.model

    # Print model info
    print(f"\nDevice: {engine.device}")
    print(f"Model dtype: {engine._model.model.dtype}")

    # Check GPU memory
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    # Create voice profile
    print("\n" + "=" * 60)
    print("Creating voice profile...")
    ref_audio = Path("tests/test_audio.wav")

    if not ref_audio.exists():
        print("No reference audio found!")
        return

    print(f"Using reference: {ref_audio}")

    profile = engine.clone_voice(
        ref_audio=str(ref_audio),
        ref_text="This is a reference audio sample.",
        name="test_profile",
    )
    print("Voice profile created!")

    # Test texts of varying length
    test_cases = [
        "Hi!",  # Very short
        "Hello there!",  # Short
        "The quick brown fox jumps over the lazy dog.",  # Medium
        "This is a longer sentence to test the generation speed of the model and see how it scales.",  # Long
    ]

    print("\n" + "=" * 60)
    print("PROFILING GENERATION")
    print("=" * 60)

    for text in test_cases:
        print(f"\n--- Text: '{text[:50]}...' ({len(text)} chars) ---")

        # Warm up GPU
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        # Access internal model for detailed profiling
        internal_model = engine._model

        # Prepare input (same as generate_voice_clone does)
        lang = profile.language
        texts = [text]
        languages = [lang]

        # Build text input
        input_texts = [internal_model._build_assistant_text(t) for t in texts]
        input_ids = internal_model._tokenize_texts(input_texts)

        # Prepare voice clone prompt
        voice_clone_prompt_dict = internal_model._prompt_items_to_voice_clone_prompt(profile.prompt_items)

        ref_ids = []
        for item in profile.prompt_items:
            if item.ref_text:
                ref_tok = internal_model._tokenize_texts([internal_model._build_ref_text(item.ref_text)])[0]
                ref_ids.append(ref_tok)
            else:
                ref_ids.append(None)

        gen_kwargs = internal_model._merge_generate_kwargs()

        # PHASE 1: Token Generation
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1_start = time.perf_counter()

        with torch.no_grad():
            talker_codes_list, _ = internal_model.model.generate(
                input_ids=input_ids,
                ref_ids=ref_ids,
                voice_clone_prompt=voice_clone_prompt_dict,
                languages=languages,
                non_streaming_mode=False,
                **gen_kwargs,
            )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1_end = time.perf_counter()
        token_gen_time = t1_end - t1_start

        # Get token count
        num_tokens = talker_codes_list[0].shape[0]
        tokens_per_sec = num_tokens / token_gen_time

        print(f"  Token generation: {token_gen_time*1000:.0f}ms | {num_tokens} tokens | {tokens_per_sec:.1f} tok/s")

        # PHASE 2: Audio Decoding
        # Prepare codes for decode (same as generate_voice_clone does)
        codes_for_decode = []
        for i, codes in enumerate(talker_codes_list):
            ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
            if ref_code_list is not None and ref_code_list[i] is not None:
                codes_for_decode.append(torch.cat([ref_code_list[i].to(codes.device), codes], dim=0))
            else:
                codes_for_decode.append(codes)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t2_start = time.perf_counter()

        with torch.no_grad():
            wavs_all, fs = internal_model.model.speech_tokenizer.decode(
                [{"audio_codes": c} for c in codes_for_decode]
            )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t2_end = time.perf_counter()
        decode_time = t2_end - t2_start

        # Calculate audio duration
        audio_duration = len(wavs_all[0]) / fs

        print(f"  Audio decode:     {decode_time*1000:.0f}ms | {audio_duration:.2f}s audio")
        print(f"  TOTAL:            {(token_gen_time + decode_time)*1000:.0f}ms")
        print(f"  Real-time factor: {audio_duration / (token_gen_time + decode_time):.2f}x")
        print(f"  Required tok/s for real-time: {num_tokens / audio_duration:.1f}")

    # Check if CUDA is being used efficiently
    print("\n" + "=" * 60)
    print("GPU ANALYSIS")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
        print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")

        # Check if model is on GPU
        for name, param in list(internal_model.model.talker.named_parameters())[:3]:
            print(f"Model param '{name}' device: {param.device}")
            break

        # Check tokenizer device
        for name, param in list(internal_model.model.speech_tokenizer.model.named_parameters())[:3]:
            print(f"Tokenizer param '{name}' device: {param.device}")
            break

if __name__ == "__main__":
    profile_tts()
