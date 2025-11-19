"""Download a small, public-domain sample audio into ./inputs/sample.wav.

This helps users and Colab runs quickly verify the pipeline end-to-end.
"""
from __future__ import annotations
from pathlib import Path
import urllib.request

SAMPLE_URL = "https://cdn.jsdelivr.net/gh/anars/blank-audio/1-second-of-silence.wav"
# Note: Replace with a more musical CC0 sample if available later.


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    out = inputs / "sample.wav"
    print(f"Downloading sample audio to {out} ...")
    urllib.request.urlretrieve(SAMPLE_URL, str(out))
    print("Done.")


if __name__ == "__main__":
    main()
