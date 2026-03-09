"""
Gradio Web Interface for Qwen3-TTS Voice Cloning System
"""

import os
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np

import config

# YouTube download support
try:
    import yt_dlp
    YOUTUBE_AVAILABLE = True
except ImportError:
    YOUTUBE_AVAILABLE = False

# Whisper STT support
try:
    import whisper
    WHISPER_AVAILABLE = True
    _whisper_model = None  # Lazy loaded
except ImportError:
    WHISPER_AVAILABLE = False
    _whisper_model = None


def get_whisper_model():
    """Get or load Whisper model (lazy loading)."""
    global _whisper_model
    if _whisper_model is None and WHISPER_AVAILABLE:
        # Use "base" model for good balance of speed/accuracy
        # Options: tiny, base, small, medium, large
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file using Whisper."""
    if not WHISPER_AVAILABLE:
        return ""

    try:
        model = get_whisper_model()
        if model is None:
            return ""

        result = model.transcribe(audio_path)
        return result.get("text", "").strip()
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

# Get ffmpeg path - check local tools folder first, then imageio-ffmpeg
FFMPEG_PATH = None
_local_ffmpeg = Path(__file__).parent.parent / "tools" / "ffmpeg-master-latest-win64-gpl" / "bin"
if _local_ffmpeg.exists() and (_local_ffmpeg / "ffmpeg.exe").exists():
    FFMPEG_PATH = str(_local_ffmpeg)
else:
    try:
        import imageio_ffmpeg
        FFMPEG_PATH = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        pass

# Add ffmpeg to PATH so Whisper can find it
if FFMPEG_PATH and FFMPEG_PATH not in os.environ.get('PATH', ''):
    os.environ['PATH'] = FFMPEG_PATH + os.pathsep + os.environ.get('PATH', '')
from src import (
    TTSEngine,
    VoiceManager,
    load_audio,
    validate_reference_audio,
    save_audio,
    get_audio_duration,
    AudioStreamer,
    analyze_audio,
    compare_audio,
    format_analysis_report,
)


# Global instances (lazy-loaded)
_engine: Optional[TTSEngine] = None
_voice_manager: Optional[VoiceManager] = None


def get_engine() -> TTSEngine:
    """Get or create TTS engine singleton."""
    global _engine
    if _engine is None:
        _engine = TTSEngine(
            model_name=config.MODEL_NAME,
            dtype=config.MODEL_DTYPE,
            use_flash_attention=config.USE_FLASH_ATTENTION,
            quantization=getattr(config, 'QUANTIZATION', None),
            use_torch_compile=getattr(config, 'USE_TORCH_COMPILE', False),
        )
    return _engine


def get_voice_manager() -> VoiceManager:
    """Get or create voice manager singleton."""
    global _voice_manager
    if _voice_manager is None:
        _voice_manager = VoiceManager(config.VOICES_DIR)
    return _voice_manager


def get_voice_choices() -> list:
    """Get list of available voices for dropdown."""
    manager = get_voice_manager()
    voices = manager.list_voices()
    return [v.name for v in voices]


def get_language_choices() -> list:
    """Get list of supported languages."""
    return config.SUPPORTED_LANGUAGES


# YouTube Download Functions

def download_youtube_audio(
    url: str,
    start_time: str = "",
    end_time: str = "",
) -> Tuple[Optional[Tuple[int, np.ndarray]], str, str]:
    """Download audio from a YouTube video and transcribe it."""
    if not YOUTUBE_AVAILABLE:
        return None, "", "Error: yt-dlp is not installed. Run: pip install yt-dlp"

    if not url or not url.strip():
        return None, "", "Error: Please enter a YouTube URL."

    url = url.strip()

    # Validate URL
    if not any(domain in url.lower() for domain in ['youtube.com', 'youtu.be']):
        return None, "", "Error: Please enter a valid YouTube URL."

    try:
        # Create temp directory for download
        temp_dir = tempfile.mkdtemp(prefix="yt_audio_")
        output_path = os.path.join(temp_dir, "audio.wav")

        # Configure yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
        }

        # Use ffmpeg from local tools or imageio-ffmpeg
        if FFMPEG_PATH:
            ydl_opts['ffmpeg_location'] = FFMPEG_PATH

        # Add time range if specified
        if start_time or end_time:
            postprocessor_args = []
            if start_time:
                postprocessor_args.extend(['-ss', start_time])
            if end_time:
                postprocessor_args.extend(['-to', end_time])
            if postprocessor_args:
                ydl_opts['postprocessor_args'] = {'ffmpeg': postprocessor_args}

        # Download audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'Unknown')

        # Find the downloaded file
        audio_file = None
        for f in os.listdir(temp_dir):
            if f.endswith('.wav'):
                audio_file = os.path.join(temp_dir, f)
                break

        if not audio_file or not os.path.exists(audio_file):
            return None, "", "Error: Failed to download audio from YouTube."

        # Load the audio
        audio_data, sr = load_audio(audio_file)

        # Transcribe the audio before cleanup
        transcription = ""
        if WHISPER_AVAILABLE:
            try:
                transcription = transcribe_audio(audio_file)
            except Exception as e:
                print(f"Transcription failed: {e}")
                transcription = ""

        # Convert to format Gradio expects
        if audio_data.dtype == np.float32:
            # Convert to int16 for Gradio
            audio_int16 = (audio_data * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data

        # Clean up temp files
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass

        duration = len(audio_data) / sr
        time_info = ""
        if start_time or end_time:
            time_info = f" (trimmed: {start_time or '0:00'} - {end_time or 'end'})"

        status = f"Downloaded: {title} ({duration:.1f}s){time_info}"
        if transcription:
            status += " - Transcription complete!"
        elif WHISPER_AVAILABLE:
            status += " - Transcription failed"
        else:
            status += " - Whisper not installed for auto-transcription"

        return (sr, audio_int16), transcription, status

    except Exception as e:
        traceback.print_exc()
        return None, "", f"Error downloading YouTube audio: {str(e)}"


# Voice Cloning Functions

def clone_voice(
    audio_input: Optional[Tuple[int, np.ndarray]],
    name: str,
    ref_text: str,
    language: str,
) -> str:
    """Clone a voice from uploaded audio."""
    if audio_input is None:
        return "Error: Please upload or record audio first."

    if not name or not name.strip():
        return "Error: Please provide a name for the voice."

    if not ref_text or not ref_text.strip():
        return "Error: Please provide a transcription of the audio."

    name = name.strip()
    ref_text = ref_text.strip()

    manager = get_voice_manager()

    # Check if voice already exists
    if manager.voice_exists(name):
        return f"Error: Voice '{name}' already exists. Delete it first or use a different name."

    try:
        # Extract audio data
        sr, audio_data = audio_input

        # Convert to float32 if needed
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                audio_data = audio_data.astype(np.float32)

        # Convert stereo to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Validate audio
        is_valid, error = validate_reference_audio(audio_data, sr)
        if not is_valid:
            return f"Error: {error}"

        # Save to temp file (model needs file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            save_audio(audio_data, sr, tmp.name)
            temp_path = tmp.name

        try:
            # Clone the voice
            engine = get_engine()
            voice_profile = engine.clone_voice(
                ref_audio=temp_path,
                ref_text=ref_text,
                name=name,
                language=language,
            )

            # Save the voice profile
            manager.save_voice(voice_profile)

            duration = get_audio_duration(audio_data, sr)
            return f"Success! Voice '{name}' cloned from {duration:.1f}s of audio."

        finally:
            # Clean up temp file
            try:
                Path(temp_path).unlink()
            except:
                pass

    except Exception as e:
        traceback.print_exc()
        return f"Error: Voice cloning failed - {str(e)}"


# Text-to-Speech Functions

def generate_speech(
    text: str,
    voice_name: str,
    language: str,
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    """Generate speech from text using a cloned voice."""
    if not text or not text.strip():
        return None, "Error: Please enter some text to synthesize."

    if not voice_name:
        return None, "Error: Please select a voice."

    manager = get_voice_manager()
    voice = manager.load_voice(voice_name)

    if voice is None:
        return None, f"Error: Voice '{voice_name}' not found. Try refreshing the voice list."

    try:
        engine = get_engine()
        result = engine.generate(
            text=text.strip(),
            voice_profile=voice,
            language=language,
        )

        # Return as (sample_rate, audio_array) for Gradio
        return (result.sample_rate, result.audio_data), f"Generated {result.duration_seconds:.2f}s of audio."

    except Exception as e:
        traceback.print_exc()
        return None, f"Error: Generation failed - {str(e)}"


# Voice Management Functions

def get_voices_table() -> list:
    """Get voices as table data."""
    manager = get_voice_manager()
    voices = manager.list_voices()

    return [
        [v.name, v.language, v.created_at[:10], v.ref_text[:50] + "..." if len(v.ref_text) > 50 else v.ref_text]
        for v in voices
    ]


def delete_voice(name: str) -> Tuple[list, str]:
    """Delete a voice profile."""
    if not name or not name.strip():
        return get_voices_table(), "Error: Please enter a voice name to delete."

    manager = get_voice_manager()

    if not manager.voice_exists(name):
        return get_voices_table(), f"Error: Voice '{name}' not found."

    manager.delete_voice(name)
    return get_voices_table(), f"Voice '{name}' deleted successfully."


def refresh_voices() -> Tuple[gr.Dropdown, list]:
    """Refresh voice list and table."""
    choices = get_voice_choices()
    table = get_voices_table()
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None), table


# Audio Analysis Functions

def analyze_audio_file(
    audio_input: Optional[Tuple[int, np.ndarray]],
) -> str:
    """Analyze an uploaded audio file."""
    if audio_input is None:
        return "Please upload an audio file to analyze."

    try:
        sr, audio_data = audio_input

        # Convert to float32 if needed
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                audio_data = audio_data.astype(np.float32)

        # Convert stereo to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Analyze
        analysis = analyze_audio(audio_data, sr)
        return format_analysis_report(analysis)

    except Exception as e:
        traceback.print_exc()
        return f"Error analyzing audio: {str(e)}"


def compare_audio_files(
    audio1_input: Optional[Tuple[int, np.ndarray]],
    audio2_input: Optional[Tuple[int, np.ndarray]],
) -> str:
    """Compare two audio files."""
    if audio1_input is None:
        return "Please upload the first audio file."
    if audio2_input is None:
        return "Please upload the second audio file."

    try:
        sr1, audio_data1 = audio1_input
        sr2, audio_data2 = audio2_input

        # Convert to float32
        for audio_data, var_name in [(audio_data1, "audio1"), (audio_data2, "audio2")]:
            if audio_data.dtype != np.float32:
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_data = audio_data.astype(np.float32) / 2147483648.0
                else:
                    audio_data = audio_data.astype(np.float32)
            if var_name == "audio1":
                audio_data1 = audio_data
            else:
                audio_data2 = audio_data

        # Convert stereo to mono
        if len(audio_data1.shape) > 1:
            audio_data1 = np.mean(audio_data1, axis=1)
        if len(audio_data2.shape) > 1:
            audio_data2 = np.mean(audio_data2, axis=1)

        # Compare
        comparison = compare_audio(audio_data1, sr1, audio_data2, sr2)

        # Format report
        lines = [
            "=" * 50,
            "AUDIO COMPARISON REPORT",
            "=" * 50,
            "",
            "--- Duration ---",
            f"  Difference: {comparison['duration_diff_seconds']:.2f} seconds",
            "",
            "--- Spectral Differences ---",
            f"  Centroid: {comparison['centroid_diff']:.3f} (0=identical)",
            f"  Bandwidth: {comparison['bandwidth_diff']:.3f} (0=identical)",
            "",
            "--- Prosody Differences ---",
            f"  Pitch Mean: {comparison['pitch_mean_diff']:.3f} (0=identical)",
            f"  Pitch Std: {comparison['pitch_std_diff']:.3f} (0=identical)",
            "",
            "--- Quality Differences ---",
            f"  RMS Energy: {comparison['rms_diff']:.3f} (0=identical)",
            f"  Dynamic Range: {comparison['dynamic_range_diff']:.3f} (0=identical)",
            "",
            "--- Overall ---",
            f"  Similarity Score: {comparison['overall_similarity']:.1%}",
            "",
            "=" * 50,
        ]
        return "\n".join(lines)

    except Exception as e:
        traceback.print_exc()
        return f"Error comparing audio: {str(e)}"


def create_app() -> gr.Blocks:
    """Create the Gradio application."""

    with gr.Blocks(
        title="Qwen3-TTS Voice Cloning",
    ) as app:

        gr.Markdown("""
        # Qwen3-TTS Voice Cloning System

        Clone voices from audio samples and generate natural speech.

        **Ethical Use Notice**: This tool is for legitimate purposes only.
        Only clone your own voice or voices you have explicit permission to use.
        """)

        with gr.Tabs():

            # Tab 1: Voice Cloning
            with gr.TabItem("Clone Voice"):
                gr.Markdown("### Create a new voice from audio")

                # YouTube Download Section
                with gr.Accordion("Download from YouTube", open=False):
                    gr.Markdown("Paste a YouTube URL to extract audio for voice cloning.")

                    yt_url = gr.Textbox(
                        label="YouTube URL",
                        placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/...",
                        max_lines=1,
                    )

                    with gr.Row():
                        yt_start = gr.Textbox(
                            label="Start Time (optional)",
                            placeholder="e.g., 0:30 or 1:25",
                            max_lines=1,
                            scale=1,
                        )
                        yt_end = gr.Textbox(
                            label="End Time (optional)",
                            placeholder="e.g., 0:45 or 1:35",
                            max_lines=1,
                            scale=1,
                        )
                        yt_fetch_btn = gr.Button("Fetch Audio", variant="secondary", scale=1)

                    yt_status = gr.Textbox(label="YouTube Status", interactive=False)

                with gr.Row():
                    with gr.Column(scale=1):
                        clone_audio = gr.Audio(
                            label="Reference Audio (3-30 seconds)",
                            type="numpy",
                            sources=["upload", "microphone"],
                        )

                    with gr.Column(scale=1):
                        clone_name = gr.Textbox(
                            label="Voice Name",
                            placeholder="e.g., MyVoice",
                            max_lines=1,
                        )
                        clone_language = gr.Dropdown(
                            label="Language",
                            choices=get_language_choices(),
                            value=config.DEFAULT_LANGUAGE,
                        )

                clone_text = gr.Textbox(
                    label="Transcription (what is being said in the audio)",
                    placeholder="Type exactly what is spoken in the audio...",
                    lines=3,
                )

                clone_btn = gr.Button("Clone Voice", variant="primary")
                clone_status = gr.Textbox(label="Status", interactive=False)

                # YouTube fetch button click - outputs: audio, transcription, status
                yt_fetch_btn.click(
                    fn=download_youtube_audio,
                    inputs=[yt_url, yt_start, yt_end],
                    outputs=[clone_audio, clone_text, yt_status],
                )

                clone_btn.click(
                    fn=clone_voice,
                    inputs=[clone_audio, clone_name, clone_text, clone_language],
                    outputs=[clone_status],
                )

                gr.Markdown("""
                **Tips:**
                - Use clear audio with minimal background noise
                - 5-10 seconds of speech works well
                - Accurate transcription improves voice quality
                - For YouTube: Use start/end times to extract just the speech portion
                """)

            # Tab 2: Text-to-Speech
            with gr.TabItem("Generate Speech"):
                gr.Markdown("### Generate speech using a cloned voice")

                with gr.Row():
                    tts_voice = gr.Dropdown(
                        label="Voice",
                        choices=get_voice_choices(),
                        value=get_voice_choices()[0] if get_voice_choices() else None,
                        scale=2,
                    )
                    tts_language = gr.Dropdown(
                        label="Language",
                        choices=get_language_choices(),
                        value=config.DEFAULT_LANGUAGE,
                        scale=1,
                    )
                    refresh_btn = gr.Button("Refresh Voices", scale=1)

                tts_text = gr.Textbox(
                    label="Text to Speak",
                    placeholder="Enter the text you want to convert to speech...",
                    lines=5,
                )

                generate_btn = gr.Button("Generate Speech", variant="primary")

                tts_output = gr.Audio(
                    label="Generated Audio",
                    type="numpy",
                    autoplay=True,
                )
                tts_status = gr.Textbox(label="Status", interactive=False)

                generate_btn.click(
                    fn=generate_speech,
                    inputs=[tts_text, tts_voice, tts_language],
                    outputs=[tts_output, tts_status],
                )

                def refresh_voice_dropdown():
                    choices = get_voice_choices()
                    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

                refresh_btn.click(
                    fn=refresh_voice_dropdown,
                    outputs=[tts_voice],
                )

                gr.Markdown("""
                **Tips:**
                - Keep sentences reasonably short for best quality
                - Use punctuation to control pauses
                - Try different languages if supported by the voice
                """)

            # Tab 3: Voice Management
            with gr.TabItem("Manage Voices"):
                gr.Markdown("### View and manage saved voices")

                voices_table = gr.Dataframe(
                    headers=["Name", "Language", "Created", "Reference Text"],
                    value=get_voices_table(),
                    interactive=False,
                )

                with gr.Row():
                    delete_name = gr.Textbox(
                        label="Voice Name to Delete",
                        placeholder="Enter voice name...",
                        scale=3,
                    )
                    delete_btn = gr.Button("Delete Voice", variant="stop", scale=1)

                manage_status = gr.Textbox(label="Status", interactive=False)

                refresh_table_btn = gr.Button("Refresh List")

                delete_btn.click(
                    fn=delete_voice,
                    inputs=[delete_name],
                    outputs=[voices_table, manage_status],
                )

                def refresh_table():
                    return get_voices_table()

                refresh_table_btn.click(
                    fn=refresh_table,
                    outputs=[voices_table],
                )

            # Tab 4: Audio Analysis
            with gr.TabItem("Analysis"):
                gr.Markdown("### Analyze audio quality metrics")

                with gr.Tabs():
                    # Sub-tab: Single Audio Analysis
                    with gr.TabItem("Analyze Audio"):
                        gr.Markdown("Upload an audio file to analyze its quality metrics.")

                        analysis_audio = gr.Audio(
                            label="Audio to Analyze",
                            type="numpy",
                            sources=["upload"],
                        )

                        analyze_btn = gr.Button("Analyze", variant="primary")

                        analysis_output = gr.Textbox(
                            label="Analysis Results",
                            lines=25,
                            interactive=False,
                        )

                        analyze_btn.click(
                            fn=analyze_audio_file,
                            inputs=[analysis_audio],
                            outputs=[analysis_output],
                        )

                    # Sub-tab: Compare Two Audio Files
                    with gr.TabItem("Compare Audio"):
                        gr.Markdown("Compare two audio files to measure their similarity.")

                        with gr.Row():
                            compare_audio1 = gr.Audio(
                                label="Audio 1 (e.g., Reference)",
                                type="numpy",
                                sources=["upload"],
                            )
                            compare_audio2 = gr.Audio(
                                label="Audio 2 (e.g., Generated)",
                                type="numpy",
                                sources=["upload"],
                            )

                        compare_btn = gr.Button("Compare", variant="primary")

                        compare_output = gr.Textbox(
                            label="Comparison Results",
                            lines=20,
                            interactive=False,
                        )

                        compare_btn.click(
                            fn=compare_audio_files,
                            inputs=[compare_audio1, compare_audio2],
                            outputs=[compare_output],
                        )

                gr.Markdown("""
                **Metrics Explained:**
                - **Spectral Centroid**: Audio brightness (higher = brighter sound)
                - **Bandwidth**: Range of frequencies present
                - **Pitch**: Fundamental frequency of speech
                - **RMS Energy**: Average loudness
                - **Dynamic Range**: Difference between loud and quiet parts
                - **SNR**: Signal-to-noise ratio estimate
                """)

        gr.Markdown("""
        ---
        **Qwen3-TTS Voice Cloning System** | Powered by [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
        """)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_port=config.GRADIO_PORT, theme=gr.themes.Soft())
