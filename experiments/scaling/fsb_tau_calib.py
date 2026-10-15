from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import torch

SCHEMA = "fsb_tau_calib/1"
TRAIN_FRAC = 0.98  # chunks drawn from the first 98% of the corpus, strictly inside the train split


def calib_chunks(data_path, n, length, seed, train_frac=TRAIN_FRAC):
    # n distinct chunks of length tokens (stride length + 1, the trainer's chunking) from the first
    # train_frac of the uint16 corpus, chosen by numpy's default_rng(seed) without replacement
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    stride = int(length) + 1
    total = len(data) // stride
    hi = int(total * train_frac)
    if n > hi:
        raise ValueError(f"fsb_tau_calib: {n} chunks of {length} tokens requested but only {hi} lie in the first "
                         f"{train_frac:.0%} of {data_path}")
    rng = np.random.default_rng(int(seed))
    ids = np.sort(rng.choice(hi, size=int(n), replace=False))
    chunks = [torch.from_numpy(data[int(c) * stride:int(c) * stride + int(length)].astype(np.int64)).view(1, -1)
              for c in ids]
    return [int(c) for c in ids], chunks


def tau_cache_path(cache_dir, ckpt_sha, quantile, n, length, seed):
    return os.path.join(cache_dir, f"fsb_tau_{str(ckpt_sha)[:16]}_q{float(quantile):g}_n{int(n)}_l{int(length)}_s{int(seed)}.pt")


def tau_sha256(tau: torch.Tensor) -> str:
    t = tau.detach().float().cpu().contiguous()
    h = hashlib.sha256(str(tuple(t.shape)).encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()


def _fsb_layers(model):
    return [m for m in model.modules() if getattr(m, "fsb_enabled", False)]


def tau_summary(model, target="fsb_tau"):
    # per FS-B layer [min, mean, max] of target (fsb_tau | fsb_tau_union) over heads, plus the overall one
    taus = torch.stack([getattr(m, target).detach().float().flatten() for m in _fsb_layers(model)]).cpu()   # (L,H)
    per = torch.stack([taus.min(1).values, taus.mean(1), taus.max(1).values], 1).tolist()
    return per, [float(taus.min()), float(taus.mean()), float(taus.max())]


@torch.no_grad()
def calibrate_or_load(model, *, ckpt_sha, quantile, n_batches, length, seed, data_path, cache_dir, device,
                      autocast_dtype=torch.bfloat16, log=print, target="fsb_tau"):
    # calibrate every FS-B layer's fsb_tau on training chunks, or load the cached tensor of the same key;
    # the cache file depends on the checkpoint and the calibration key only, so both targets share it
    from src.fsattn.fsb_tree import fsb_calibrate_tau
    layers = _fsb_layers(model)
    if not layers:
        raise ValueError("fsb_tau_calib: the model has no FS-B layer")
    if not quantile > 0:
        raise ValueError(f"fsb_tau_calib: quantile must be > 0 (got {quantile})")
    if target not in ("fsb_tau", "fsb_tau_union"):
        raise ValueError(f"fsb_tau_calib: target {target!r} (fsb_tau | fsb_tau_union)")
    key = dict(schema=SCHEMA, ckpt_sha256=str(ckpt_sha), quantile=float(quantile), n_batches=int(n_batches),
               length=int(length), seed=int(seed), data=os.path.abspath(data_path), train_frac=TRAIN_FRAC)
    os.makedirs(cache_dir, exist_ok=True)
    path = tau_cache_path(cache_dir, ckpt_sha, quantile, n_batches, length, seed)
    L, H = len(layers), int(layers[0].fsb_tau.numel())
    if os.path.isfile(path):
        obj = torch.load(path, map_location="cpu")
        meta = dict(obj["meta"])
        got = {k: meta.get(k) for k in key}
        if got != key:
            raise RuntimeError(f"fsb_tau_calib: {path} was calibrated for another key: {got} != {key}")
        tau = obj["tau"].float()
        if tuple(tau.shape) != (L, H):
            raise RuntimeError(f"fsb_tau_calib: {path} holds tau of shape {tuple(tau.shape)}, the model has (L,H) = {(L, H)}")
        if tau_sha256(tau) != meta.get("tau_sha256"):
            raise RuntimeError(f"fsb_tau_calib: {path} is corrupt (tau sha256 mismatch)")
        for m, t in zip(layers, tau):
            buf = getattr(m, target)
            buf.copy_(t.to(buf.device, buf.dtype))
        meta.update(cache_file=path, loaded_from_cache=True, target=target)
        log(f"[fsb-tau] loaded {target} (L,H) = {(L, H)} from {path} (q {key['quantile']:g}, {key['n_batches']} chunks x "
            f"{key['length']} tokens, seed {key['seed']}, sha {meta['tau_sha256'][:16]})")
        return meta
    t0 = time.time()
    ids, chunks = calib_chunks(data_path, n_batches, length, seed)
    dev = torch.device(device)
    ac = torch.autocast(device_type=dev.type, dtype=autocast_dtype, enabled=autocast_dtype is not None)
    with ac:
        res = fsb_calibrate_tau(model, [c.to(dev) for c in chunks], quantile, target=target)
    assert res is not None and tuple(res["tau"].shape) == (L, H), (None if res is None else tuple(res["tau"].shape), (L, H))
    tau = res["tau"].float().contiguous()
    meta = dict(key, chunk_ids=ids, tau_sha256=tau_sha256(tau), n_cells_per_head=int(res["n_cells"]),
                admit_frac_calib_layer=res["admit_frac"].mean(1).tolist(),
                admit_frac_calib_mean=float(res["admit_frac"].mean()),
                time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), seconds=round(time.time() - t0, 2))
    # atomic write, then the re-read must reproduce the computed tensor exactly
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save({"tau": tau, "meta": meta}, tmp)
    os.replace(tmp, path)
    back = torch.load(path, map_location="cpu")["tau"]
    if not torch.equal(back, tau):
        raise RuntimeError(f"fsb_tau_calib: {path} re-read differs from the computed tau")
    held = torch.stack([getattr(m, target).detach().float().cpu() for m in layers])
    if not torch.equal(held, tau):
        raise RuntimeError(f"fsb_tau_calib: the model's {target} differs from the calibrated tensor")
    meta.update(cache_file=path, loaded_from_cache=False, target=target)
    log(f"[fsb-tau] calibrated {target} (L,H) = {(L, H)} at q {float(quantile):g} on {n_batches} training chunks x {length} "
        f"tokens (seed {seed}, chunk ids {ids}) in {meta['seconds']}s; admitted fraction on the calibration cells "
        f"{meta['admit_frac_calib_mean']:.3f}; saved {path} (sha {meta['tau_sha256'][:16]})")
    return meta
