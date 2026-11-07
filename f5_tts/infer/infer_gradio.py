"""Lightweight Gradio interface for IndicF5 inference.

The model is loaded lazily on the first synthesis request so importing this
module or starting the UI does not download model weights or require a GPU.
"""

import os
import tempfile
from functools import lru_cache

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel


DEFAULT_MODEL_ID = os.environ.get("INDICF5_MODEL_ID", "ai4bharat/IndicF5")
SAMPLE_RATE = 24000


@lru_cache(maxsize=1)
def load_model(model_id: str = DEFAULT_MODEL_ID):
    """Load and cache IndicF5 only when the first request needs it."""
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(model, "eval"):
        model.eval()
    return model


def _to_audio_array(audio):
    """Convert model output to a Gradio-compatible floating-point array."""
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio)
    if np.issubdtype(audio.dtype, np.integer):
        scale = max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / scale
    else:
        audio = audio.astype(np.float32, copy=False)
    return np.squeeze(audio)


def synthesize(ref_audio, ref_text, gen_text, model_id):
    """Generate speech using the selected IndicF5 checkpoint."""
    if not ref_audio:
        raise gr.Error("Upload a reference audio clip to continue.")
    if not (gen_text or "").strip():
        raise gr.Error("Enter the text you want IndicF5 to speak.")

    model_id = (model_id or DEFAULT_MODEL_ID).strip()
    model = load_model(model_id)
    try:
        with torch.inference_mode():
            audio = model(
                gen_text.strip(),
                ref_audio_path=ref_audio,
                ref_text=(ref_text or "").strip(),
            )
    except Exception as exc:
        raise gr.Error(f"IndicF5 synthesis failed: {exc}") from exc

    waveform = _to_audio_array(audio)
    if waveform.size == 0:
        raise gr.Error("IndicF5 returned an empty audio result.")
    return SAMPLE_RATE, waveform


def load_text_file(file_path):
    if not file_path:
        return ""
    with open(file_path, encoding="utf-8") as text_file:
        return text_file.read()


with gr.Blocks(title="IndicF5 Text-to-Speech") as app:
    gr.Markdown(
        """# IndicF5 Text-to-Speech

Upload a short reference recording, provide its transcript when available, and
enter the text to synthesize. The model accepts Indic-language text; no
language selector or language-specific input filter is applied here.
"""
    )

    with gr.Row():
        model_id = gr.Textbox(
            label="IndicF5 model ID or local model directory",
            value=DEFAULT_MODEL_ID,
            info="Defaults to ai4bharat/IndicF5. You can also enter a local Hugging Face model directory.",
        )

    ref_audio = gr.Audio(label="Reference audio", type="filepath")
    with gr.Row():
        ref_text = gr.Textbox(
            label="Reference transcript",
            lines=3,
            placeholder="Enter the words spoken in the reference audio, if known.",
        )
        ref_text_file = gr.File(label="Load reference transcript (.txt)", file_types=[".txt"], type="filepath")
    ref_text_file.upload(load_text_file, inputs=ref_text_file, outputs=ref_text)

    with gr.Row():
        gen_text = gr.Textbox(label="Text to synthesize", lines=8, scale=4)
        gen_text_file = gr.File(label="Load text (.txt)", file_types=[".txt"], type="filepath", scale=1)
    gen_text_file.upload(load_text_file, inputs=gen_text_file, outputs=gen_text)

    generate = gr.Button("Synthesize", variant="primary")
    audio_output = gr.Audio(label="Generated speech")
    generate.click(synthesize, inputs=[ref_audio, ref_text, gen_text, model_id], outputs=audio_output)


@click.command()
@click.option("--host", "-H", default=None, help="Host interface to bind to.")
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on.")
@click.option("--share", "-s", is_flag=True, help="Create a Gradio share link.")
@click.option("--inbrowser", "-i", is_flag=True, help="Open the app in a browser.")
def main(host, port, share, inbrowser):
    """Run the IndicF5 Gradio interface."""
    app.queue().launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=inbrowser,
    )


if __name__ == "__main__":
    main()
