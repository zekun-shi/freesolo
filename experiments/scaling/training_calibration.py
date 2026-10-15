"""Frozen second-forest calibration for weights-only continued pretraining."""

import hashlib
import json
from pathlib import Path


def validate_union_training(args):
    path = getattr(args, "fsb_union_tau_file", None)
    quantile = getattr(args, "fsb_union_tau_quantile", None)
    if path is None and quantile is None:
        return
    if not path or quantile is None or not 0 < quantile < 1:
        raise ValueError("Union CPT calibration needs both a file and a quantile in (0, 1)")
    if not (args.fsb and args.fsb_tree == "forest" and args.fsb_policy == "fixed"
            and (args.fsb_union_width or 0) > 0 and (args.fsb_read_top or 0) > 0):
        raise ValueError("Union CPT calibration requires fixed two-forest FS-B with a final read budget")
    if not args.init_from or args.resume:
        raise ValueError("Union CPT calibration supports weights-only --init_from, not --resume")


def load_union_training(model, args):
    validate_union_training(args)
    path = getattr(args, "fsb_union_tau_file", None)
    if path is None:
        return None
    from experiments.scaling.fsb_release import load_union_calibration
    digest = hashlib.sha256()
    with (Path(args.init_from) / "model.safetensors").open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    load_union_calibration(model, path, checkpoint_sha256=digest.hexdigest(),
                           quantile=args.fsb_union_tau_quantile)
    return json.loads(Path(path).read_text())


def save_union_training(model, payload, directory):
    """Preserve fixed thresholds and their initial-checkpoint provenance."""
    import torch
    from experiments.scaling.fsb_tau_calib import tau_sha256
    layers = [m for m in model.modules() if getattr(m, "fsb_enabled", False)]
    tau = torch.stack([m.fsb_tau_union.detach().float().cpu() for m in layers])
    if tau_sha256(tau) != payload["tau_sha256"]:
        raise RuntimeError("Frozen union thresholds changed during CPT")
    path = Path(directory) / "training_union_calibration.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
