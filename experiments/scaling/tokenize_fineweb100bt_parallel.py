#!/usr/bin/env python
import os, sys, glob, time, multiprocessing as mp
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import pyarrow.parquet as pq

MODEL   = "fla-hub/transformer-1.3B-100B"
SRC     = "./data/parquet/sample/100BT"
OUTDIR  = "./data/fineweb100bt_llama_parts"
FINAL   = "./data/fineweb_edu_llama32k_100bt.bin"
NWORKERS = int(os.environ.get("NWORKERS", "40"))
TEXTCOL = "text"
FLUSH   = 16_000_000
BATCH   = 2000

_tok = None
def get_tok():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(MODEL)
        _tok.model_max_length = int(1e9)
    return _tok

def worker(arg):
    idx, files = arg
    outp = os.path.join(OUTDIR, f"part_{idx:03d}.bin")
    if os.path.exists(outp):
        return idx, os.path.getsize(outp) // 2, outp, True
    tok = get_tok(); sep = tok.eos_token_id
    tmp = outp + ".tmp"; total = 0; buf = []; bufn = 0
    with open(tmp, "wb") as f:
        for fp in files:
            for batch in pq.ParquetFile(fp).iter_batches(batch_size=BATCH, columns=[TEXTCOL]):
                for s in batch.column(0).to_pylist():
                    if not s:
                        continue
                    ids = tok.encode(s, add_special_tokens=False)
                    ids.append(sep)
                    a = np.asarray(ids, dtype=np.uint16)
                    buf.append(a); bufn += a.size; total += a.size
                    if bufn >= FLUSH:
                        np.concatenate(buf).tofile(f); buf = []; bufn = 0
        if buf:
            np.concatenate(buf).tofile(f)
    os.replace(tmp, outp)
    return idx, total, outp, False

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "*.parquet")))
    if not files:
        print(f"[err] no parquet under {SRC}", flush=True); sys.exit(1)
    shards = [[] for _ in range(NWORKERS)]
    for i, fp in enumerate(files):
        shards[i % NWORKERS].append(fp)
    shards = [(i, s) for i, s in enumerate(shards) if s]
    print(f"[cfg] {len(files)} parquet files -> {len(shards)} workers | model={MODEL} | out={FINAL}", flush=True)
    tok0 = get_tok()
    print(f"[warm] tokenizer={type(tok0).__name__} vocab={tok0.vocab_size} eos={tok0.eos_token_id}", flush=True)
    t0 = time.time(); results = []
    with mp.Pool(len(shards)) as pool:
        for idx, tot, outp, skipped in pool.imap_unordered(worker, shards):
            results.append((idx, tot))
            done = sum(t for _, t in results); dt = time.time() - t0
            tag = "skip" if skipped else "done"
            print(f"[{tag}] part {idx:03d} {tot/1e9:.2f}B | cum {done/1e9:.2f}B | "
                  f"{dt/60:.1f}min | {done/dt/1e6:.1f}M tok/s", flush=True)
    grand = sum(t for _, t in results)
    print(f"[all-parts] {grand/1e9:.3f}B tokens in {(time.time()-t0)/60:.1f}min; concatenating...", flush=True)
    parts = sorted(glob.glob(os.path.join(OUTDIR, "part_*.bin")))
    ftmp = FINAL + ".tmp"
    with open(ftmp, "wb") as out:
        for p in parts:
            with open(p, "rb") as pf:
                while True:
                    b = pf.read(64 * 1024 * 1024)
                    if not b:
                        break
                    out.write(b)
    os.replace(ftmp, FINAL)
    sz = os.path.getsize(FINAL)
    print(f"[final] {FINAL} {sz} bytes = {sz//2/1e9:.3f}B tokens. DONE.", flush=True)

if __name__ == "__main__":
    main()
