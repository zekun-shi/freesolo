"""Retrieve a hidden number directly from Hugging Face.

Run on one CUDA GPU after installing FreeSolo:
    python examples/hf_generate.py --model zekun-shi/FreeSolo-B-341M --output generation.json

Equivalent installed command used for A100 GPU validation:
    freesolo-generate --model zekun-shi/FreeSolo-B-341M --output generation.json

Expected answer: 5437923. The JSON records the exact Hub revision and generated tokens.
Private previews require `hf auth login` or HF_TOKEN in the environment.
"""

from experiments.scaling.hub import generate_main


if __name__ == "__main__":
    generate_main()
