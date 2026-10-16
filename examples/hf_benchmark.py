"""Measure the paper's FS-B speed from a Hugging Face download on one idle A100-40GB.

    python examples/hf_benchmark.py --model zekun-shi/FreeSolo-B-341M \
        --lengths 65536,131072 --output hf-benchmark.json

BF16, batch 1, median of 3, warm-up 1, 256 decode steps, no memory cap.
Reference: prefill[graph] 78.6/72.7 Ktok/s; decode[graph] 205.3/206.7 tok/s.
Weights and calibration are downloaded together; no training corpus is needed.
"""

from experiments.scaling.hub import benchmark_main


if __name__ == "__main__":
    benchmark_main()
