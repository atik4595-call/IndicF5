# IndicF5 Gradio UI

Run the web interface from the repository root:

```bash
python -m f5_tts.infer.infer_gradio
```

The interface loads `ai4bharat/IndicF5` the first time you synthesize speech,
then reuses the loaded model for subsequent requests. To use a different
Hugging Face model ID or a local model directory, set `INDICF5_MODEL_ID` before
launching, or edit the model field in the UI:

```bash
INDICF5_MODEL_ID=/path/to/model python -m f5_tts.infer.infer_gradio --host 127.0.0.1 --port 7860
```

The interface takes reference audio, its transcript (optional), and generation
text. It does not impose a language allowlist. The existing IndicF5 package,
model/tokenizer code, prompt assets, and dependency files are left in place.
