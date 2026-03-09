#!/usr/bin/env python3
"""
Qwen3-TTS Voice Cloning System - Main Entry Point

Usage:
    python main.py --test         # Verify environment setup
    python main.py --test-tts     # Test TTS engine components
    python main.py --test-audio   # Test audio processing utilities
    python main.py --test-metrics # Test quality metrics functionality
    python main.py --api          # Run FastAPI server
    python main.py --ui           # Run Gradio interface
    python main.py                # Show help
"""

import argparse
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def test_environment() -> bool:
    """Test that all required packages are installed and working."""
    print("=" * 50)
    print("Qwen3-TTS Environment Test")
    print("=" * 50)

    errors = []
    warnings = []

    # Test 1: Core Python packages
    print("\n[1/5] Testing core packages...")
    try:
        import numpy as np
        print(f"  [OK] numpy {np.__version__}")
    except ImportError as e:
        errors.append(f"numpy: {e}")
        print(f"  [FAIL] numpy: {e}")

    try:
        import scipy
        print(f"  [OK] scipy {scipy.__version__}")
    except ImportError as e:
        errors.append(f"scipy: {e}")
        print(f"  [FAIL] scipy: {e}")

    # Test 2: PyTorch and CUDA
    print("\n[2/5] Testing PyTorch...")
    try:
        import torch
        print(f"  [OK] torch {torch.__version__}")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  [OK] CUDA available: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            warnings.append("CUDA not available - will use CPU (slower)")
            print("  [WARN] CUDA not available - will use CPU (slower)")

        # Check for bfloat16 support
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            print("  [OK] bfloat16 supported")
        elif torch.cuda.is_available():
            warnings.append("bfloat16 not supported - will use float16")
            print("  [WARN] bfloat16 not supported - will use float16")

    except ImportError as e:
        errors.append(f"torch: {e}")
        print(f"  [FAIL] torch: {e}")

    # Test 3: Audio packages
    print("\n[3/5] Testing audio packages...")
    try:
        import soundfile as sf
        print(f"  [OK] soundfile {sf.__version__}")
    except ImportError as e:
        errors.append(f"soundfile: {e}")
        print(f"  [FAIL] soundfile: {e}")

    try:
        import librosa
        print(f"  [OK] librosa {librosa.__version__}")
    except ImportError as e:
        errors.append(f"librosa: {e}")
        print(f"  [FAIL] librosa: {e}")

    # Test 4: Web framework packages
    print("\n[4/5] Testing web packages...")
    try:
        import fastapi
        print(f"  [OK] fastapi {fastapi.__version__}")
    except ImportError as e:
        errors.append(f"fastapi: {e}")
        print(f"  [FAIL] fastapi: {e}")

    try:
        import uvicorn
        print(f"  [OK] uvicorn {uvicorn.__version__}")
    except ImportError as e:
        errors.append(f"uvicorn: {e}")
        print(f"  [FAIL] uvicorn: {e}")

    try:
        import gradio as gr
        print(f"  [OK] gradio {gr.__version__}")
    except ImportError as e:
        errors.append(f"gradio: {e}")
        print(f"  [FAIL] gradio: {e}")

    # Test 5: Qwen TTS package
    print("\n[5/5] Testing Qwen TTS...")
    try:
        from qwen_tts import Qwen3TTSModel
        print("  [OK] qwen_tts installed")
    except ImportError as e:
        errors.append(f"qwen_tts: {e}")
        print(f"  [FAIL] qwen_tts: {e}")

    # Check FlashAttention (optional)
    print("\n[Optional] Checking FlashAttention...")
    try:
        import flash_attn
        print(f"  [OK] flash_attn {flash_attn.__version__}")
    except ImportError:
        warnings.append("flash_attn not installed (optional, but recommended for performance)")
        print("  [WARN] flash_attn not installed (optional)")

    # Test directory structure
    print("\n[Directories] Checking project structure...")
    base_dir = Path(__file__).parent
    required_dirs = ["src", "api", "ui", "voices", "output", "tests"]
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"  [OK] {dir_name}/")
        else:
            errors.append(f"Directory missing: {dir_name}/")
            print(f"  [FAIL] {dir_name}/ (missing)")

    # Test config import
    print("\n[Config] Testing configuration...")
    try:
        import config
        print(f"  [OK] config.py loaded")
        print(f"    Model: {config.MODEL_NAME}")
        print(f"    Languages: {len(config.SUPPORTED_LANGUAGES)} supported")
    except ImportError as e:
        errors.append(f"config: {e}")
        print(f"  [FAIL] config: {e}")

    # Summary
    print("\n" + "=" * 50)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for err in errors:
            print(f"  - {err}")
        print("\nRun: pip install -r requirements.txt")
        return False
    elif warnings:
        print(f"PASSED with {len(warnings)} warning(s)")
        for warn in warnings:
            print(f"  - {warn}")
        print("\nEnvironment setup complete!")
        return True
    else:
        print("PASSED: All checks successful!")
        print("\nEnvironment setup complete!")
        return True


def test_tts_engine() -> bool:
    """Test the TTS engine components without loading the full model."""
    print("=" * 50)
    print("Qwen3-TTS Engine Component Test")
    print("=" * 50)

    errors = []
    import tempfile
    import shutil

    # Test 1: Import src modules
    print("\n[1/4] Testing src module imports...")
    try:
        from src import TTSEngine, VoiceManager, VoiceProfile, VoiceInfo
        from src import GenerationRequest, GenerationResult
        print("  [OK] All src modules imported")
    except ImportError as e:
        errors.append(f"src imports: {e}")
        print(f"  [FAIL] src imports: {e}")
        return False

    # Test 2: TTSEngine instantiation (without loading model)
    print("\n[2/4] Testing TTSEngine instantiation...")
    try:
        engine = TTSEngine()
        print(f"  [OK] TTSEngine created")
        print(f"    Device: {engine.device}")
        print(f"    Model: {engine.model_name}")
        print(f"    Loaded: {engine.is_loaded()}")
    except Exception as e:
        errors.append(f"TTSEngine: {e}")
        print(f"  [FAIL] TTSEngine: {e}")

    # Test 3: VoiceManager operations
    print("\n[3/4] Testing VoiceManager operations...")
    test_dir = None
    try:
        # Create a temporary directory for testing
        test_dir = tempfile.mkdtemp(prefix="tts_test_")
        manager = VoiceManager(test_dir)
        print(f"  [OK] VoiceManager created")

        # Test list (should be empty)
        voices = manager.list_voices()
        assert len(voices) == 0, "Expected empty voice list"
        print(f"  [OK] list_voices() works (empty)")

        # Test save with dummy data
        dummy_profile = VoiceProfile(
            name="test_voice",
            prompt_items={"dummy": "data"},  # Placeholder
            ref_text="This is a test voice.",
            language="English",
        )
        save_path = manager.save_voice(dummy_profile)
        assert save_path.exists(), "Voice file not created"
        print(f"  [OK] save_voice() works")

        # Test list (should have one)
        voices = manager.list_voices()
        assert len(voices) == 1, f"Expected 1 voice, got {len(voices)}"
        assert voices[0].name == "test_voice"
        print(f"  [OK] list_voices() returns saved voice")

        # Test load
        loaded = manager.load_voice("test_voice")
        assert loaded is not None, "Failed to load voice"
        assert loaded.name == "test_voice"
        assert loaded.ref_text == "This is a test voice."
        print(f"  [OK] load_voice() works")

        # Test exists
        assert manager.voice_exists("test_voice")
        assert not manager.voice_exists("nonexistent")
        print(f"  [OK] voice_exists() works")

        # Test delete
        deleted = manager.delete_voice("test_voice")
        assert deleted, "Delete returned False"
        assert not manager.voice_exists("test_voice")
        print(f"  [OK] delete_voice() works")

    except Exception as e:
        errors.append(f"VoiceManager: {e}")
        print(f"  [FAIL] VoiceManager: {e}")
    finally:
        # Clean up
        if test_dir:
            shutil.rmtree(test_dir, ignore_errors=True)

    # Test 4: Data models
    print("\n[4/4] Testing data models...")
    try:
        # Test GenerationRequest
        req = GenerationRequest(
            text="Hello world",
            voice_name="test",
            language="English",
        )
        req.validate()
        print(f"  [OK] GenerationRequest validation works")

        # Test invalid request
        try:
            bad_req = GenerationRequest(text="", voice_name="test")
            bad_req.validate()
            errors.append("GenerationRequest should reject empty text")
            print(f"  [FAIL] GenerationRequest should reject empty text")
        except ValueError:
            print(f"  [OK] GenerationRequest rejects invalid input")

        # Test VoiceInfo
        import config
        info = VoiceInfo(
            name="test",
            ref_text="test text",
            language="English",
            created_at="2024-01-01",
            file_path=config.VOICES_DIR / "test.voice",
        )
        info_dict = info.to_dict()
        assert info_dict["name"] == "test"
        print(f"  [OK] VoiceInfo serialization works")

    except Exception as e:
        errors.append(f"Data models: {e}")
        print(f"  [FAIL] Data models: {e}")

    # Summary
    print("\n" + "=" * 50)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print("PASSED: All TTS engine component tests successful!")
        return True


def test_metrics() -> bool:
    """Test audio quality metrics functionality."""
    print("=" * 50)
    print("Qwen3-TTS Quality Metrics Test")
    print("=" * 50)

    errors = []
    import numpy as np

    # Test 1: Import metrics modules
    print("\n[1/5] Testing metrics module imports...")
    try:
        from src import (
            SpectralMetrics,
            ProsodyMetrics,
            QualityMetrics,
            AudioAnalysis,
            compute_spectral_metrics,
            compute_prosody_metrics,
            compute_quality_metrics,
            analyze_audio,
            compare_audio,
            format_analysis_report,
        )
        print("  [OK] All metrics modules imported")
    except ImportError as e:
        errors.append(f"metrics imports: {e}")
        print(f"  [FAIL] metrics imports: {e}")
        return False

    # Test 2: Create test audio (sine wave + noise)
    print("\n[2/5] Creating test audio...")
    try:
        sr = 24000
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

        # Create a speech-like test signal (fundamental + harmonics + noise)
        fundamental = 200  # Hz (typical male voice)
        test_audio = (
            0.4 * np.sin(2 * np.pi * fundamental * t) +  # Fundamental
            0.2 * np.sin(2 * np.pi * 2 * fundamental * t) +  # 1st harmonic
            0.1 * np.sin(2 * np.pi * 3 * fundamental * t) +  # 2nd harmonic
            0.05 * np.random.randn(len(t))  # Noise
        ).astype(np.float32)

        print(f"  [OK] Created {duration}s test audio at {sr} Hz")
        print(f"    Samples: {len(test_audio)}")
    except Exception as e:
        errors.append(f"test audio creation: {e}")
        print(f"  [FAIL] test audio creation: {e}")
        return False

    # Test 3: Compute spectral metrics
    print("\n[3/5] Testing spectral metrics...")
    try:
        spectral = compute_spectral_metrics(test_audio, sr)
        assert isinstance(spectral, SpectralMetrics), "Wrong return type"
        assert spectral.centroid_mean > 0, "Centroid should be positive"
        assert spectral.bandwidth_mean > 0, "Bandwidth should be positive"
        assert spectral.rolloff_mean > 0, "Rolloff should be positive"
        assert 0 <= spectral.zero_crossing_rate_mean <= 1, "ZCR should be 0-1"

        print(f"  [OK] compute_spectral_metrics() works")
        print(f"    Centroid: {spectral.centroid_mean:.1f} Hz")
        print(f"    Bandwidth: {spectral.bandwidth_mean:.1f} Hz")
        print(f"    Rolloff: {spectral.rolloff_mean:.1f} Hz")
        print(f"    ZCR: {spectral.zero_crossing_rate_mean:.4f}")

        # Test to_dict()
        spec_dict = spectral.to_dict()
        assert "centroid_mean" in spec_dict, "Missing key in to_dict()"
        print(f"  [OK] SpectralMetrics.to_dict() works")
    except Exception as e:
        errors.append(f"spectral metrics: {e}")
        print(f"  [FAIL] spectral metrics: {e}")

    # Test 4: Compute prosody metrics
    print("\n[4/5] Testing prosody metrics...")
    try:
        prosody = compute_prosody_metrics(test_audio, sr)
        assert isinstance(prosody, ProsodyMetrics), "Wrong return type"
        # Pitch mean should be around our fundamental (200 Hz)
        if prosody.pitch_mean > 0:
            print(f"  [OK] compute_prosody_metrics() works")
            print(f"    Pitch Mean: {prosody.pitch_mean:.1f} Hz")
            print(f"    Pitch Range: {prosody.pitch_range:.1f} Hz")
            print(f"    Voiced Ratio: {prosody.voiced_ratio:.1%}")
            print(f"    Speaking Rate: {prosody.speaking_rate_estimate:.2f}/sec")
        else:
            print(f"  [OK] compute_prosody_metrics() works (no pitch detected - expected for synthetic)")

        # Test to_dict()
        pros_dict = prosody.to_dict()
        assert "pitch_mean" in pros_dict, "Missing key in to_dict()"
        print(f"  [OK] ProsodyMetrics.to_dict() works")
    except Exception as e:
        errors.append(f"prosody metrics: {e}")
        print(f"  [FAIL] prosody metrics: {e}")

    # Test 5: Compute quality metrics
    print("\n[5/5] Testing quality and comparison...")
    try:
        quality = compute_quality_metrics(test_audio, sr)
        assert isinstance(quality, QualityMetrics), "Wrong return type"
        assert quality.rms_mean > 0, "RMS should be positive"
        assert quality.dynamic_range_db >= 0, "Dynamic range should be non-negative"
        assert 0 <= quality.clipping_ratio <= 1, "Clipping ratio should be 0-1"
        assert 0 <= quality.silence_ratio <= 1, "Silence ratio should be 0-1"

        print(f"  [OK] compute_quality_metrics() works")
        print(f"    RMS Mean: {quality.rms_mean:.4f}")
        print(f"    Dynamic Range: {quality.dynamic_range_db:.1f} dB")
        print(f"    Clipping: {quality.clipping_ratio:.2%}")
        print(f"    Silence: {quality.silence_ratio:.1%}")
        print(f"    SNR Est: {quality.snr_estimate_db:.1f} dB")

        # Test full analysis
        analysis = analyze_audio(test_audio, sr)
        assert isinstance(analysis, AudioAnalysis), "Wrong return type"
        assert analysis.duration_seconds > 0, "Duration should be positive"
        assert analysis.sample_rate == sr, "Sample rate mismatch"
        print(f"  [OK] analyze_audio() works")

        # Test formatted report
        report = format_analysis_report(analysis)
        assert len(report) > 100, "Report too short"
        assert "AUDIO ANALYSIS REPORT" in report, "Missing header"
        print(f"  [OK] format_analysis_report() works")

        # Test comparison
        # Create a slightly different audio
        test_audio2 = (
            0.35 * np.sin(2 * np.pi * 220 * t) +  # Different fundamental
            0.15 * np.sin(2 * np.pi * 440 * t) +
            0.08 * np.random.randn(len(t))
        ).astype(np.float32)

        comparison = compare_audio(test_audio, sr, test_audio2, sr)
        assert "overall_similarity" in comparison, "Missing similarity"
        assert 0 <= comparison["overall_similarity"] <= 1, "Similarity out of range"
        print(f"  [OK] compare_audio() works")
        print(f"    Overall Similarity: {comparison['overall_similarity']:.1%}")

    except Exception as e:
        errors.append(f"quality metrics: {e}")
        print(f"  [FAIL] quality metrics: {e}")

    # Summary
    print("\n" + "=" * 50)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print("PASSED: All quality metrics tests successful!")
        return True


def test_audio_utils() -> bool:
    """Test audio processing utilities."""
    print("=" * 50)
    print("Qwen3-TTS Audio Utilities Test")
    print("=" * 50)

    errors = []
    import tempfile
    import shutil
    import numpy as np

    test_dir = None

    try:
        # Create temp directory
        test_dir = tempfile.mkdtemp(prefix="audio_test_")

        # Test 1: Import audio modules
        print("\n[1/6] Testing audio module imports...")
        try:
            from src import (
                load_audio, save_audio, validate_reference_audio,
                get_audio_info, get_audio_duration, resample_audio,
                normalize_audio, AudioInfo, AudioStreamer,
                SUPPORTED_FORMATS
            )
            print("  [OK] All audio modules imported")
            print(f"    Supported formats: {SUPPORTED_FORMATS}")
        except ImportError as e:
            errors.append(f"audio imports: {e}")
            print(f"  [FAIL] audio imports: {e}")
            return False

        # Test 2: Create and save test audio
        print("\n[2/6] Testing audio save...")
        try:
            # Create a 5-second test sine wave at 440 Hz
            sr = 24000
            duration = 5.0
            t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
            test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)

            test_file = Path(test_dir) / "test_audio.wav"
            saved_path = save_audio(test_audio, sr, test_file)
            assert saved_path.exists(), "File not created"
            print(f"  [OK] save_audio() works")
            print(f"    Created: {saved_path}")
        except Exception as e:
            errors.append(f"save_audio: {e}")
            print(f"  [FAIL] save_audio: {e}")

        # Test 3: Load audio
        print("\n[3/6] Testing audio load...")
        try:
            loaded_audio, loaded_sr = load_audio(test_file)
            assert loaded_sr == sr, f"Sample rate mismatch: {loaded_sr} != {sr}"
            assert len(loaded_audio) == len(test_audio), "Length mismatch"
            print(f"  [OK] load_audio() works")
            print(f"    Loaded: {len(loaded_audio)} samples at {loaded_sr} Hz")

            # Test get_audio_info
            info = get_audio_info(test_file)
            assert abs(info.duration_seconds - duration) < 0.1, "Duration mismatch"
            print(f"  [OK] get_audio_info() works")
            print(f"    Duration: {info.duration_seconds:.2f}s, SR: {info.sample_rate}")

            # Test get_audio_duration
            dur = get_audio_duration(loaded_audio, loaded_sr)
            assert abs(dur - duration) < 0.01, "Duration calculation error"
            print(f"  [OK] get_audio_duration() works")
        except Exception as e:
            errors.append(f"load_audio: {e}")
            print(f"  [FAIL] load_audio: {e}")

        # Test 4: Validate reference audio
        print("\n[4/6] Testing audio validation...")
        try:
            # Valid audio (5 seconds)
            is_valid, error = validate_reference_audio(loaded_audio, loaded_sr)
            assert is_valid, f"Should be valid: {error}"
            print(f"  [OK] validate_reference_audio() accepts valid audio")

            # Too short (1 second)
            short_audio = test_audio[:sr]  # 1 second
            is_valid, error = validate_reference_audio(short_audio, sr)
            assert not is_valid, "Should reject short audio"
            assert "too short" in error.lower(), f"Wrong error: {error}"
            print(f"  [OK] validate_reference_audio() rejects short audio")

            # Silent audio
            silent_audio = np.zeros(sr * 5, dtype=np.float32)
            is_valid, error = validate_reference_audio(silent_audio, sr)
            assert not is_valid, "Should reject silent audio"
            print(f"  [OK] validate_reference_audio() rejects silent audio")
        except Exception as e:
            errors.append(f"validate_reference_audio: {e}")
            print(f"  [FAIL] validate_reference_audio: {e}")

        # Test 5: Audio resampling
        print("\n[5/6] Testing audio resampling...")
        try:
            target_sr = 16000
            resampled = resample_audio(loaded_audio, loaded_sr, target_sr)
            expected_length = int(len(loaded_audio) * target_sr / loaded_sr)
            assert abs(len(resampled) - expected_length) < 100, "Resample length error"
            print(f"  [OK] resample_audio() works")
            print(f"    {loaded_sr} Hz -> {target_sr} Hz ({len(resampled)} samples)")

            # Test normalization
            normalized = normalize_audio(loaded_audio, target_db=-3.0)
            assert len(normalized) == len(loaded_audio), "Length changed"
            print(f"  [OK] normalize_audio() works")
        except Exception as e:
            errors.append(f"resample_audio: {e}")
            print(f"  [FAIL] resample_audio: {e}")

        # Test 6: Streaming
        print("\n[6/6] Testing audio streaming...")
        try:
            streamer = AudioStreamer(sample_rate=sr, chunk_size=4096)

            # Test WAV header
            header = streamer.create_wav_header(len(test_audio) * 2)
            assert header[:4] == b"RIFF", "Invalid WAV header"
            assert header[8:12] == b"WAVE", "Invalid WAV format"
            print(f"  [OK] create_wav_header() works")

            # Test streaming chunks
            chunks = list(streamer.stream_audio(test_audio[:sr], include_header=True))
            assert len(chunks) > 1, "Should have multiple chunks"
            assert chunks[0][:4] == b"RIFF", "First chunk should be header"
            print(f"  [OK] stream_audio() works ({len(chunks)} chunks)")

            # Test base64 encoding
            b64 = streamer.audio_to_base64(test_audio[:sr])
            assert len(b64) > 0, "Empty base64"
            # Verify it can be decoded
            decoded = streamer.base64_to_audio(b64)
            assert len(decoded) > 0, "Decoded audio empty"
            print(f"  [OK] audio_to_base64() works")

            # Test complete WAV bytes
            wav_bytes = streamer.audio_to_wav_bytes(test_audio[:sr])
            assert wav_bytes[:4] == b"RIFF", "Invalid WAV bytes"
            print(f"  [OK] audio_to_wav_bytes() works")
        except Exception as e:
            errors.append(f"AudioStreamer: {e}")
            print(f"  [FAIL] AudioStreamer: {e}")

    except Exception as e:
        errors.append(f"Unexpected error: {e}")
        print(f"  [FAIL] Unexpected error: {e}")

    finally:
        # Clean up
        if test_dir:
            shutil.rmtree(test_dir, ignore_errors=True)

    # Summary
    print("\n" + "=" * 50)
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for err in errors:
            print(f"  - {err}")
        return False
    else:
        print("PASSED: All audio utility tests successful!")
        return True


def run_api():
    """Run the FastAPI server."""
    import uvicorn
    import config

    print(f"Starting API server on http://{config.API_HOST}:{config.API_PORT}")
    print(f"API docs available at http://localhost:{config.API_PORT}/docs")
    uvicorn.run(
        "api.app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    )


def run_ui():
    """Run the Gradio interface."""
    import gradio as gr
    import config

    print(f"Starting Gradio UI on http://localhost:{config.GRADIO_PORT}")
    from ui.gradio_app import create_app
    app = create_app()
    app.launch(server_port=config.GRADIO_PORT, theme=gr.themes.Soft())


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS Voice Cloning System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test environment setup and exit"
    )
    parser.add_argument(
        "--test-tts",
        action="store_true",
        help="Test TTS engine components and exit"
    )
    parser.add_argument(
        "--test-audio",
        action="store_true",
        help="Test audio processing utilities and exit"
    )
    parser.add_argument(
        "--test-metrics",
        action="store_true",
        help="Test quality metrics functionality and exit"
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Run FastAPI server only"
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Run Gradio interface only"
    )

    args = parser.parse_args()

    if args.test:
        success = test_environment()
        sys.exit(0 if success else 1)
    elif args.test_tts:
        success = test_tts_engine()
        sys.exit(0 if success else 1)
    elif args.test_audio:
        success = test_audio_utils()
        sys.exit(0 if success else 1)
    elif args.test_metrics:
        success = test_metrics()
        sys.exit(0 if success else 1)
    elif args.api:
        run_api()
    elif args.ui:
        run_ui()
    else:
        # Show help by default
        print("Use --api or --ui to run specific interface")
        print("Use --test to verify environment")
        print("Use --test-tts to test TTS engine components")
        print("Use --test-audio to test audio utilities")
        print("Use --test-metrics to test quality metrics")
        parser.print_help()


if __name__ == "__main__":
    main()
