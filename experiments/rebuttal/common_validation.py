"""Matched held-out token CE for the recovered FS-L continuation seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments/scaling')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--packets', type=Path, required=True)
    p.add_argument('--seed-label', type=int, choices=[42, 43, 44], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=320)
    a = p.parse_args()
    assert 1 <= a.limit <= 320 and not a.output.exists()
    os.environ.update(FS_FUSED_ATTN='1', FS_STRICT_CAUSAL='1', FS_PRUNE_KERNEL='1',
        FSL_LINEAR_CACHE='1', FS_PRUNE_POOL='64', FS_PRUNE_WINDOW='8',
        FS_PRUNE_BLKCHUNK='16', TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0')
    import numpy as np
    import torch
    from safetensors import safe_open
    from experiments.rebuttal.bounded_likelihood import completion_logprobs
    from experiments.scaling.bench_efficiency import serve_kwargs
    from experiments.scaling.longctx_infer import load_model
    from experiments.scaling.model import MODEL_CONFIGS

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.manual_seed(42)
    checkpoint_sha = sha(a.checkpoint / 'model.safetensors')
    assert checkpoint_sha == a.checkpoint_sha256
    metadata = json.loads((a.packets / 'manifest.json').read_text())
    receipt = json.loads((a.packets / 'validation.bin.json').read_text())
    data = a.packets / 'validation.bin'
    assert metadata['corpus_sha256'] == '9ae37aba95ce006afa73954fc1bb7ad7c8843c9d41c291078a62386f0b5cdddb'
    assert receipt['corpus_sha256'] == metadata['corpus_sha256']
    assert receipt['plan_sha256'] == metadata['plan_sha256']
    assert receipt['file'] == 'validation.bin' and receipt['split'] == 'validation'
    assert receipt['bytes'] == data.stat().st_size == 320 * 4097 * 2
    assert sha(data) == receipt['sha256']
    tokens = np.fromfile(data, dtype=np.uint16).reshape(320, 4097)
    assert int(tokens.max()) < 32000
    with safe_open(str(a.checkpoint / 'model.safetensors'), framework='pt') as weights:
        stored = set(weights.keys())
        assert weights.get_slice('tok_emb.weight').get_shape() == [32000, 1536]
    MODEL_CONFIGS['800m']['vocab_size'] = 32000
    model = load_model(str(a.checkpoint), config='800m', device='cuda',
        max_seq_len=4096, **serve_kwargs('sparse', 0.0))
    expected = set(model.state_dict())
    assert not (stored - expected) and (expected - stored) <= {'lm_head.weight'}
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()
    assert model.count_params() == 742863360
    assert all(p.dtype == torch.float32 for p in model.parameters())
    result = dict(status='running', seed_label=a.seed_label,
        checkpoint=str(a.checkpoint), checkpoint_sha256=checkpoint_sha,
        source_commit=(ROOT / 'SOURCE_COMMIT').read_text().strip(), argv=sys.argv,
        script_sha256=sha(Path(__file__)), bounded_adapter_sha256=sha(ROOT / 'experiments/rebuttal/bounded_likelihood.py'),
        data_receipt=receipt, corpus_manifest_sha256=sha(a.packets / 'manifest.json'),
        precision='FP32 weights, BF16 autocast', implementation='fully bounded FS-L P64/W8/K4',
        model_config=dict(MODEL_CONFIGS['800m'], max_seq_len=4096),
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        started_at=time.time(), sequences=[])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index, ids in enumerate(tokens[:a.limit]):
            logp = completion_logprobs(model, ids.astype(np.int64), 4096, chunk_size=2048)
            assert logp.numel() == 4096
            nll = -float(logp.double().sum().item())
            assert math.isfinite(nll) and nll >= 0
            result['sequences'].append(dict(index=index, tokens=4096, nll=nll,
                input_sha256=hashlib.sha256(ids.tobytes()).hexdigest()))
            if (index + 1) % 20 == 0 or index + 1 == a.limit:
                a.output.with_suffix('.progress.json').write_text(json.dumps(result, indent=2) + '\n')
                print(json.dumps(dict(completed=index + 1, planned=a.limit)), flush=True)
    total = sum(x['tokens'] for x in result['sequences'])
    ce = sum(x['nll'] for x in result['sequences']) / total
    result.update(status='complete', n_sequences=a.limit, n_tokens=total,
        cross_entropy=ce, perplexity=math.exp(ce), completed_at=time.time())
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(seed=a.seed_label, cross_entropy=ce, perplexity=math.exp(ce), n_tokens=total)), flush=True)


if __name__ == '__main__':
    main()
