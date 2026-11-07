"""Gradio web interface for IndicF5 inference."""

import os
from functools import lru_cache

import click
import gradio as gr

from f5_tts.api import DEFAULT_MODEL_ID, IndicF5


DEFAULT_MODEL = os.environ.get("INDICF5_MODEL_ID", DEFAULT_MODEL_ID)


@lru_cache(maxsize=2)
def load_tts(model_id):
    """Load once per model ID, on demand, and reuse for later generations."""
    return IndicF5(model_id)


def synthesize(ref_audio, ref_text, gen_text, model_id, seed):
    if not ref_audio:
        raise gr.Error("Upload reference audio first.")
    if not (gen_text or "").strip():
        raise gr.Error("Enter the text you want to synthesize.")
    try:
        tts = load_tts((model_id or DEFAULT_MODEL).strip())
        waveform, sample_rate, _ = tts.infer(
            ref_audio,
            ref_text or "",
            gen_text,
            seed=int(seed) if seed is not None else -1,
        )
        return sample_rate, waveform
    except Exception as exc:
        raise gr.Error(f"IndicF5 synthesis failed: {exc}") from exc


def read_text_file(file_path):
    if not file_path:
        return ""
    with open(file_path, encoding="utf-8") as text_file:
        return text_file.read()


with gr.Blocks(title="IndicF5 Text-to-Speech") as app:
    gr.Markdown(
        """# IndicF5 Text-to-Speech

Provide reference audio and its transcript, then enter the text to synthesize.
IndicF5 handles the supported Indic languages through its own pretrained
pipeline; this interface does not add a language allowlist or replace its
model-specific tokenization.
"""
    )
    model_id = gr.Textbox(
        label="Model ID or local model directory",
        value=DEFAULT_MODEL,
        info="Defaults to ai4bharat/IndicF5. Weights load on the first synthesis request and are reused.",
    )
    ref_audio = gr.Audio(label="Reference audio", type="filepath")
    with gr.Row():
        ref_text = gr.Textbox(label="Reference transcript", lines=3, scale=4)
        ref_text_file = gr.File(label="Load transcript (.txt)", file_types=[".txt"], type="filepath", scale=1)
    ref_text_file.upload(read_text_file, inputs=ref_text_file, outputs=ref_text)

    with gr.Row():
        gen_text = gr.Textbox(label="Text to synthesize", lines=8, scale=4)
        gen_text_file = gr.File(label="Load text (.txt)", file_types=[".txt"], type="filepath", scale=1)
    gen_text_file.upload(read_text_file, inputs=gen_text_file, outputs=gen_text)
    seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)

    generate = gr.Button("Synthesize", variant="primary")
    audio_output = gr.Audio(label="Generated speech")
    generate.click(synthesize, inputs=[ref_audio, ref_text, gen_text, model_id, seed], outputs=audio_output)


@click.command()
@click.option("--host", "-H", default=None, help="Host interface to bind to.")
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on.")
@click.option("--share", "-s", is_flag=True, help="Create a Gradio share link.")
@click.option("--inbrowser", "-i", is_flag=True, help="Open the app in a browser.")
def main(host, port, share, inbrowser):
    app.queue().launch(server_name=host, server_port=port, share=share, inbrowser=inbrowser)


if __name__ == "__main__":
    main()
