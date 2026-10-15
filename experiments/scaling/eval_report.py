"""Completion checks for saved RULER reports."""

import argparse
import json
import math
from pathlib import Path


def result_complete(path, *, task, length, checkpoint, variant, data, limit):
    try:
        with Path(data).open() as stream:
            available = sum(bool(line.strip()) for line in stream)
        expected = min(limit, available) if limit > 0 else available
        report = json.loads(Path(path).read_text())
        meta = report["meta"]
        result = report["results"][task]
        score = result["score"]
        return (
            expected > 0
            and meta["length"] == length
            and Path(meta["ckpt"]).resolve() == Path(checkpoint).resolve()
            and meta["variant"] == variant
            and result["status"] == "ok"
            and result["n"] == expected
            and isinstance(score, (float, int))
            and not isinstance(score, bool)
            and math.isfinite(score)
            and 0 <= score <= 100
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--task", required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--variant", choices=["FS", "FS-L", "FS-B"], required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = vars(parser.parse_args())
    raise SystemExit(0 if result_complete(**args) else 1)


if __name__ == "__main__":
    main()
