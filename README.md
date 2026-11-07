# IndicF5

IndicF5 is a multilingual text-to-speech model for Indian languages. The
package keeps the original `f5_tts/` module paths, model code, prompt assets,
and language-specific model behavior. Inference uses the released
`ai4bharat/IndicF5` Hugging Face model; F5-TTS's Chinese pinyin conversion is
not substituted for IndicF5's own tokenizer/pipeline.

## Install

Use Python 3.10 or later. For Python 3.13, install a matching PyTorch and
torchaudio build for the runtime (including the desired CUDA build in Colab),
then install this project:

```bash
pip install -r requirements.txt
pip install -e .
```

The Python 3.13 dependency markers select a NumPy 2.x and PyTorch 2.6+
compatible baseline. PyTorch wheels depend on platform and CUDA; for Colab,
keep its preinstalled PyTorch build when it satisfies the version requirement,
and ensure `torch` and `torchaudio` versions match. The repository has not been
runtime-tested on every Colab accelerator/runtime combination.

## Gradio web UI

Start the UI from the repository root:

```bash
python -m f5_tts.infer.infer_gradio --host 0.0.0.0 --port 7860
```

Model weights are loaded at the first synthesis request, then cached for reuse.
Use the model field for a local model directory or another compatible model ID.

## Command-line inference

```bash
python -m f5_tts.infer.infer_cli \
  --ref_audio prompts/PAN_F_HAPPY_00001.wav \
  --ref_text "Reference transcript" \
  --gen_text "Text to synthesize" \
  --output_file samples/output.wav
```

Or provide UTF-8 input from a text file with `--gen_file`. Use `--help` for all
options. For programmatic inference, import `IndicF5` from `f5_tts.api`.

## Python API

```python
from f5_tts.api import IndicF5

model = IndicF5()
audio = model(
    "নমস্কার, কেমন আছেন?",
    ref_audio_path="prompts/PAN_F_HAPPY_00001.wav",
    ref_text="Reference audio transcript",
)
```

A supplied reference transcript avoids loading an additional ASR model. To
transcribe audio automatically, use the model's documented native API and
install its corresponding ASR dependencies; this repository's Gradio form
keeps the transcript explicit/optional rather than silently downloading an
extra speech-recognition model.

## Languages and tokenizer

Use Indic-language text supported by the selected IndicF5 checkpoint. The UI
does not impose a restrictive language allowlist. Existing tokenizer/model
files, prompt paths, and required language dependencies are preserved.
