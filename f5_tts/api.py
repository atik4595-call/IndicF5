"""Convenient inference API for the pretrained IndicF5 model.

IndicF5's released checkpoint exposes its native inference pipeline through
Hugging Face ``AutoModel``. Keeping that pipeline intact preserves its language
handling and model-specific tokenizer rather than substituting F5-TTS's Chinese
pinyin tokenizer.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel

from f5_tts.infer.utils_infer import target_sample_rate


DEFAULT_MODEL_ID = "ai4bharat/IndicF5"


@lru_cache(maxsize=2)
def _load_model(model_id):
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(model, "eval"):
        model.eval()
    return model


def _as_float_audio(audio):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    if hasattr(audio, "audio"):
        audio = audio.audio
    audio = np.asarray(audio)
    if np.issubdtype(audio.dtype, np.integer):
        limits = np.iinfo(audio.dtype)
        audio = audio.astype(np.float32) / max(abs(limits.min), limits.max)
    else:
        audio = audio.astype(np.float32, copy=False)
    return np.squeeze(audio)


class IndicF5:
    """Load and run the official IndicF5 Hugging Face model."""

    def __init__(self, model_id=DEFAULT_MODEL_ID):
        self.model_id = str(model_id)
        self.model = _load_model(self.model_id)
        self.target_sample_rate = target_sample_rate

    @torch.inference_mode()
    def infer(self, ref_file, ref_text, gen_text, *, seed=-1, file_wave=None):
        """Synthesize ``gen_text`` conditioned on reference audio and transcript."""
        if not ref_file:
            raise ValueError("A reference audio file is required.")
        if not Path(ref_file).is_file():
            raise FileNotFoundError(f"Reference audio not found: {ref_file}")
        if not (gen_text or "").strip():
            raise ValueError("Text to synthesize cannot be empty.")

        if seed is not None and seed >= 0:
            torch.manual_seed(int(seed))

        audio = self.model(
            gen_text.strip(),
            ref_audio_path=str(ref_file),
            ref_text=(ref_text or "").strip(),
        )
        waveform = _as_float_audio(audio)
        if waveform.size == 0:
            raise RuntimeError("IndicF5 returned an empty audio result.")

        if file_wave is not None:
            destination = Path(file_wave)
            destination.parent.mkdir(parents=True, exist_ok=True)
            sf.write(destination, waveform, self.target_sample_rate)
        return waveform, self.target_sample_rate, None

    def __call__(self, gen_text, ref_audio_path, ref_text="", *, seed=-1):
        waveform, _, _ = self.infer(ref_audio_path, ref_text, gen_text, seed=seed)
        return waveform


# Keep the former import name available to existing callers of this IndicF5
# distribution. New code should use the model's actual name, ``IndicF5``.
F5TTS = IndicF5
