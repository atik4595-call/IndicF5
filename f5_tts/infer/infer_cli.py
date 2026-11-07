"""Command-line inference for IndicF5."""

import argparse
from pathlib import Path

import soundfile as sf

from f5_tts.api import DEFAULT_MODEL_ID, IndicF5


def build_parser():
    parser = argparse.ArgumentParser(description="Synthesize speech with IndicF5.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Hugging Face model ID or local model directory.")
    parser.add_argument("--ref_audio", required=True, help="Reference audio file.")
    parser.add_argument("--ref_text", default="", help="Transcript of the reference audio.")
    text = parser.add_mutually_exclusive_group(required=True)
    text.add_argument("--gen_text", help="Text to synthesize.")
    text.add_argument("--gen_file", help="UTF-8 text file to synthesize.")
    parser.add_argument("--output_file", default="output.wav", help="Path for the generated WAV file.")
    parser.add_argument("--seed", type=int, default=-1, help="Optional reproducible seed; negative selects random.")
    return parser


def main():
    args = build_parser().parse_args()
    gen_text = args.gen_text
    if args.gen_file:
        gen_text = Path(args.gen_file).read_text(encoding="utf-8")

    tts = IndicF5(args.model)
    waveform, sample_rate, _ = tts.infer(
        args.ref_audio,
        args.ref_text,
        gen_text,
        seed=args.seed,
    )
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, waveform, sample_rate)
    print(f"Saved generated audio to {output_path}")


if __name__ == "__main__":
    main()
