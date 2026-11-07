from setuptools import find_packages, setup


setup(
    name="f5_tts",
    version="0.2.0",
    author="AI4Bharat contributors",
    description="IndicF5: high-quality text-to-speech for Indian languages",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AI4Bharat/IndicF5",
    packages=find_packages(include=["f5_tts", "f5_tts.*"]),
    install_requires=[
        "accelerate>=0.33.0",
        "cached_path",
        "click",
        "gradio>=4.44",
        "jieba",
        "librosa",
        "matplotlib",
        "numpy<=1.26.4; python_version < '3.13'",
        "numpy>=2.1; python_version >= '3.13'",
        "pydub",
        "pypinyin",
        "safetensors",
        "soundfile",
        "torch>=2.0.0; python_version < '3.13'",
        "torch>=2.6.0; python_version >= '3.13'",
        "torchaudio>=2.0.0; python_version < '3.13'",
        "torchaudio>=2.6.0; python_version >= '3.13'",
        "transformers<4.50",
    ],
    python_requires=">=3.10",
)
