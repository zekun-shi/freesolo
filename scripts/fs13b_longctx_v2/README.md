# FS-B-1.3B long-context v2 re-runs (2026-10-23)

Re-runs of the 22 FS-B-1.3B HELMET/BABILong cells that exhausted their 40 GiB
budget under the phase-1 campaign harness, on the current main inference stack
per the "scale13b" follow-up: lean chunked prefill + per-step graph decode.

**Track record (A100-40GB probe, this tree, FP32, `FSB_DECODE_WHOLE=0`):**
8k 9.86 GiB, 64k 25.0 GiB OK; 131k OOM at 36.5/39.4 GiB. The 64k cells below
run on protozk; the 128k cells need more memory and are only listed here.

## What is pinned

- Model code: this tree (branch `zekun/fs13b-longctx-eval`), deployed unmodified
  as a `git archive` snapshot to
  `/home/aiops/shizk/fs13b_kernel_probe_20261023/freesolo-<sha>`.
- Checkpoint: `/home/aiops/shizk/fs_longctx_evals_20261016/ckpts/FreeSolo-B-1.3B`
  (`model.safetensors` sha256 must equal the union-calibration binding;
  `config.json` declares `weights_sha256`
  422df2e136242680af48cb6349265132b126cc8fad7ba353dba2516393ee2098).
- Campaign harness (pre-existing, precedes this branch; sha256 of the files in
  use recorded 2026-10-23):
  - `harness/helmet_run.py`    08c2723b654395866327aa98065950659e33b8d30bd9f68bdae6c75d26608909
  - `harness/helmet_model.py`  100af854a8e8468d3be8bdeca869186236134d3b2d26ae9f2605f1e9c082f796
  - `harness/babilong_eval.py` da3924b32ac67591e7f5e00cdac71aad0f1a17a46e39fbd4375612c76979fb49
  - controllers: copied into this branch (see headers of `run_*_v2.sh`).
- HELMET: princeton-nlp/HELMET pinned `aeadacc` (as named by the campaign
  adapter docstring), offline data under `$W/helmet/HELMET`.
- Python envs (campaign-pinned): `physics-attn-venv` (torch 2.5.1+cu121, py3.11).

## Env deltas vs the phase-1 campaign runs (all recorded here)

1. `FS_EVAL_CODE_ROOT` / `--code-root` point at the tracked snapshot instead of
   `$W/code/rel/freesolo` (the phase-1 pinned release).
2. `FSB_DECODE_WHOLE=0` — the whole-step CUDA-graph wrapper raises
   `CUDA error: capturing stream has unjoined work` in `physics-attn-venv`
   (torch 2.5.1+cu121) on protozk; per-step graphs replay the same kernels and
   values. Files a handoff note to the kernel owners.
3. `FS_SOURCE_GIT=<branch sha>` is exported for receipt trails.
4. Phase-1 partial outputs (`results/helmet/fsb13b/*_{64k,128k}`,
   `results/babilong/fsb13b/qa*_{64k,128k}.jsonl`) are relocated to
   `results/{helmet,babilong}/_partials_v1eager/` before v2 launch, never
   deleted; the rules each controller enforces (.DONE markers / summary grep)
   regenerate destinations atomically.

## Launch plan

- 64k (11 cells): protozk GPUs 4-7 (0/1 tenants, 2/3 third-party memory —
  untouched). One controller per GPU; HELMET 8 cells then BABILong 3.
- 128k (11 cells): blocked on memory; candidate host 10.77.190.1 (idle GB200,
  shared-team grant; aarch64 port of venv+harness required) — requires owner
  approval before use. `cells_*_128k.txt` are staged here for that port.
