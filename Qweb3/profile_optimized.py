"""
Try to optimize TTS generation by reducing sync overhead.
"""
import time
import torch
from pathlib import Path
import os

# Set environment variables for better CUDA performance
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["TORCH_USE_CUDA_DSA"] = "0"

# Enable optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

def profile_tts():
    from src.tts_engine import TTSEngine
    import config

    print("=" * 60)
    print("OPTIMIZED TTS PROFILING")
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
        print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        print(f"TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")

    # Create voice profile
    print("\n" + "=" * 60)
    print("Creating voice profile...")
    ref_audio = Path("tests/test_audio.wav")

    if not ref_audio.exists():
        print("No reference audio found!")
        return

    profile = engine.clone_voice(
        ref_audio=str(ref_audio),
        ref_text="This is a reference audio sample.",
        name="test_profile",
    )
    print("Voice profile created!")

    # Warm up with a few generations
    print("\nWarming up GPU...")
    for _ in range(3):
        with torch.no_grad():
            engine.generate("Warm up text.", profile)
    torch.cuda.synchronize()
    print("Warmup complete!")

    # Test texts
    test_cases = [
        "Hello!",
        "Hello there!",
        "The quick brown fox jumps over the lazy dog.",
    ]

    print("\n" + "=" * 60)
    print("PROFILING (after warmup)")
    print("=" * 60)

    for text in test_cases:
        print(f"\n--- Text: '{text[:50]}' ({len(text)} chars) ---")

        times = []
        for run in range(3):
            torch.cuda.synchronize()
            t_start = time.perf_counter()

            with torch.no_grad():
                result = engine.generate(text, profile)

            torch.cuda.synchronize()
            t_end = time.perf_counter()
            times.append(t_end - t_start)

        avg_time = sum(times) / len(times)
        audio_duration = result.duration_seconds
        rtf = audio_duration / avg_time

        print(f"  Avg time: {avg_time*1000:.0f}ms | Audio: {audio_duration:.2f}s | RTF: {rtf:.2f}x")
        print(f"  Runs: {[f'{t*1000:.0f}ms' for t in times]}")

    # Now try with non_streaming_mode=True which might be faster
    print("\n" + "=" * 60)
    print("TESTING non_streaming_mode=True")
    print("=" * 60)

    for text in test_cases:
        print(f"\n--- Text: '{text[:50]}' ({len(text)} chars) ---")

        internal_model = engine._model
        lang = profile.language
        texts = [text]
        languages = [lang]

        input_texts = [internal_model._build_assistant_text(t) for t in texts]
        input_ids = internal_model._tokenize_texts(input_texts)
        voice_clone_prompt_dict = internal_model._prompt_items_to_voice_clone_prompt(profile.prompt_items)

        ref_ids = []
        for item in profile.prompt_items:
            if item.ref_text:
                ref_tok = internal_model._tokenize_texts([internal_model._build_ref_text(item.ref_text)])[0]
                ref_ids.append(ref_tok)
            else:
                ref_ids.append(None)

        gen_kwargs = internal_model._merge_generate_kwargs()

        times = []
        for run in range(3):
            torch.cuda.synchronize()
            t_start = time.perf_counter()

            with torch.no_grad():
                talker_codes_list, _ = internal_model.model.generate(
                    input_ids=input_ids,
                    ref_ids=ref_ids,
                    voice_clone_prompt=voice_clone_prompt_dict,
                    languages=languages,
                    non_streaming_mode=True,  # Try non-streaming mode
                    **gen_kwargs,
                )

            torch.cuda.synchronize()
            t_end = time.perf_counter()
            times.append(t_end - t_start)

        num_tokens = talker_codes_list[0].shape[0]
        avg_time = sum(times) / len(times)
        tok_per_sec = num_tokens / avg_time

        print(f"  Avg time: {avg_time*1000:.0f}ms | Tokens: {num_tokens} | {tok_per_sec:.1f} tok/s")
        print(f"  Runs: {[f'{t*1000:.0f}ms' for t in times]}")


if __name__ == "__main__":
    profile_tts()
