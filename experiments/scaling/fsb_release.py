"""Export and load portable FS-B union-forest calibration thresholds."""

import argparse
import json
from pathlib import Path

import torch

from experiments.scaling.fsb_tau_calib import tau_sha256

SCHEMA = "freesolo_union_calibration/1"
FIELDS = {"schema", "checkpoint_sha256", "quantile", "tau_sha256", "tau", "calibration"}


def export_calibration(cache_path, output):
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    meta = cache["meta"]
    tau = cache["tau"].detach().float().cpu().contiguous()
    digest = tau_sha256(tau)
    if digest != meta["tau_sha256"] or not torch.isfinite(tau).all():
        raise ValueError("Calibration cache is corrupt")
    if tau.ndim != 2 or min(tau.shape) == 0:
        raise ValueError("Calibration must have one threshold per layer and head")
    payload = {
        "schema": SCHEMA,
        "checkpoint_sha256": meta["ckpt_sha256"],
        "quantile": meta["quantile"],
        "tau_sha256": digest,
        "tau": tau.tolist(),
        "calibration": {key: meta[key] for key in
                        ("n_batches", "length", "seed", "train_frac", "chunk_ids")},
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return payload


@torch.no_grad()
def load_union_calibration(model, path, *, checkpoint_sha256, quantile):
    payload = json.loads(Path(path).read_text())
    if set(payload) != FIELDS or payload["schema"] != SCHEMA:
        raise ValueError("Unsupported union calibration format")
    if payload["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("Union calibration belongs to a different checkpoint")
    if payload["quantile"] != quantile or not 0 < quantile < 1:
        raise ValueError("Union calibration quantile does not match the model")
    layers = [module for module in model.modules() if getattr(module, "fsb_enabled", False)]
    if not layers:
        raise ValueError("The model has no FS-B layers")
    tau = torch.tensor(payload["tau"], dtype=torch.float32)
    shape = (len(layers), layers[0].fsb_tau_union.numel())
    if tuple(tau.shape) != shape or not torch.isfinite(tau).all():
        raise ValueError(f"Union calibration must contain finite thresholds with shape {shape}")
    if tau_sha256(tau) != payload["tau_sha256"]:
        raise ValueError("Union calibration checksum does not match")
    if any(tuple(module.fsb_tau_union.shape) != (shape[1],) for module in layers):
        raise ValueError("FS-B layers have inconsistent head counts")
    for module, row in zip(layers, tau):
        module.fsb_tau_union.copy_(row.to(module.fsb_tau_union))
    return {"tau_sha256": payload["tau_sha256"], "quantile": quantile,
            "checkpoint_sha256": checkpoint_sha256, "calibration": payload["calibration"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="trusted calibration cache produced by fsb_tau_calib")
    parser.add_argument("--output", required=True, help="new portable JSON file; never overwrites an existing file")
    args = parser.parse_args()
    payload = export_calibration(args.cache, args.output)
    print(f"Saved calibration: {args.output}; threshold SHA256: {payload['tau_sha256']}")


if __name__ == "__main__":
    main()
