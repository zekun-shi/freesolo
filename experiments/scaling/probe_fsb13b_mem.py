import json, os, sys, time, traceback
CODE = "/home/aiops/shizk/fs13b_kernel_probe_20261023/freesolo"
CKPT = "/home/aiops/shizk/fs_longctx_evals_20261016/ckpts/FreeSolo-B-1.3B"
OUT = f"/home/aiops/shizk/fs13b_kernel_probe_20261023/probe_results_{os.environ.get('TAG','run')}.json"
sys.path.insert(0, CODE)
import torch
import eval_longbench as elb

class A: pass
a = A()
a.architecture = "FS-B"
a.config = "1.3b_fs"
a.checkpoint = CKPT
a.pool = 64
a.weight_dtype = "float32"
a.fsb_union_tau_file = os.path.join(CKPT, "union_calibration.json")

results = {"tag": os.environ.get("TAG", "run"), "runs": []}
torch.set_grad_enabled(False)
def save():
    results["env"] = {k: v for k, v in os.environ.items() if k.startswith(("FS_", "FSB_", "FSL_"))}
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)

t0 = time.perf_counter()
model, info = elb.load_fs(a)
torch.cuda.synchronize()
results["load_seconds"] = round(time.perf_counter() - t0, 1)
results["device"] = torch.cuda.get_device_name()
results["params_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
print(f"LOADED peak={results['params_peak_gib']:.2f} GiB in {results['load_seconds']}s", flush=True)
save()

torch.manual_seed(0)
for L in (8192, 65536, 131072):
    ids = torch.randint(100, 31999, (L,))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        gen = elb.generate_fs(model, ids, 16, a, 1)
        torch.cuda.synchronize()
        rec = dict(prompt_len=L, ok=True, peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3),
                   seconds=round(time.perf_counter() - t0, 1), out_len=int(gen.numel() - L))
        print(f"L={L} OK peak={rec['peak_gib']:.2f} GiB wall={rec['seconds']}s", flush=True)
        del gen
    except Exception as e:
        rec = dict(prompt_len=L, ok=False, error=f"{type(e).__name__}: {e}"[:300],
                   peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3),
                   seconds=round(time.perf_counter() - t0, 1))
        traceback.print_exc()
        print(f"L={L} FAIL {rec['error'][:200]}", flush=True)
        results["runs"].append(rec)
        save()
        break  # CUDA state sticky after capture failure; stop cleanly
    results["runs"].append(rec)
    del ids
    save()
print("PROBE_DONE", flush=True)
save()
