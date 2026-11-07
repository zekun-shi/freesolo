"""Matched ordinary-cache timing of fully trained rebuttal controls.

Each invocation measures one checkpoint on one exclusive GPU. The existing
attention kernels remain enabled. This is a common cached execution protocol;
the separate optimized CUDA-graph serving measurements remain in Appendix D.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def put(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def gpu_process_ids(gpu_uuid):
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], text=True)
    return [int(row.split(',')[1]) for row in rows.splitlines()
            if row.split(',')[0].strip() == gpu_uuid]


def bind_exclusive_cuda_process(gpu_uuid, initialize_context):
    # NVML reports host PIDs, which need not equal os.getpid() inside the pod.
    assert not gpu_process_ids(gpu_uuid), 'GPU already has a compute process'
    initialize_context()
    owners = gpu_process_ids(gpu_uuid)
    assert len(owners) == 1, ('CUDA context must be the sole GPU process', owners)
    return owners[0]


def require_exclusive_gpu(gpu_uuid, owner_pid):
    actual = gpu_process_ids(gpu_uuid)
    assert actual == [owner_pid], ('GPU occupancy changed', actual, owner_pid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--arm', choices=['fs', 'nope', 'swan', 'dense_lr'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lengths', type=int, nargs='+', default=[4096,8192,16384,32768,65536,131072])
    parser.add_argument('--steps', type=int, default=257)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    assert args.steps >= 257 and args.repeats >= 3
    args.output.mkdir(exist_ok=False)
    assert sha(args.checkpoint/'model.safetensors') == args.checkpoint_sha256
    source_manifest = json.loads((args.source/'SOURCE_MANIFEST.json').read_text())
    assert all(sha(args.source/name)==digest for name,digest in source_manifest.items())
    sys.path[:0] = [str(args.source), str(args.source/'experiments/scaling')]
    import torch
    from experiments.scaling.model import MODEL_CONFIGS
    from experiments.scaling.bench_efficiency import ATTN_FLAGS, serve_kwargs
    from experiments.scaling.longctx_infer import load_model, _lm_head_fp32, extend_rope_cache
    from experiments.scaling.baseline_cache import make_baseline_caches, baseline_cached_generate
    from experiments.scaling.kv_cache import make_caches, _trunk_cached, cached_generate

    for key,value in ATTN_FLAGS.items():
        os.environ[key] = value
    os.environ.update(FS_FUSED_ATTN='1', FS_PRUNE_KERNEL='0', FS_STRICT_CAUSAL='1',
                      FSL_LINEAR_CACHE='0', KV_PREALLOC='1')
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(4)
    gpu_uuid = subprocess.check_output(['nvidia-smi', '-i', os.environ.get('CUDA_VISIBLE_DEVICES','0'),
        '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    assert gpu_uuid.startswith('GPU-') and '\n' not in gpu_uuid

    def initialize_context():
        marker = torch.empty(1, device='cuda')
        torch.cuda.synchronize()
        del marker

    owner_pid = bind_exclusive_cuda_process(gpu_uuid, initialize_context)

    def exclusive():
        require_exclusive_gpu(gpu_uuid, owner_pid)

    exclusive()
    config = MODEL_CONFIGS['340m'].copy()
    config['vocab_size'] = 32000
    MODEL_CONFIGS['340m'] = config
    baseline = args.arm in ('nope','swan')
    if args.arm=='nope':
        attn_type, kwargs = 'standard', dict(use_rope=False)
    elif args.arm=='swan':
        attn_type, kwargs = 'swan', dict(window_size=512, swan_scale_base=8192., swan_rope_theta=1e6)
    else:
        attn_type, kwargs = 'fsattn', serve_kwargs('dense')
        kwargs.update(lr_dense_control=args.arm=='dense_lr', fs_fused_attn=1)
    # Released safetensors omit the output-head alias tied to tok_emb.weight.
    # load_model's check_params_loaded still rejects every unloaded independent parameter.
    model = load_model(str(args.checkpoint), config='340m', attn_type=attn_type,
                       max_seq_len=4096, strict=False, device='cuda', **kwargs).eval()
    # Keep positional buffers in fp32, as in the released evaluator.
    for parameter in model.parameters():
        parameter.data = parameter.data.to(torch.bfloat16)
    if not baseline:
        extend_rope_cache(model, max(args.lengths)+args.steps+1)

    def caches_for(length, steps):
        if baseline:
            return make_baseline_caches(model, length+steps)
        caches = make_caches(model, reserve=steps, linear=False)
        for cache in caches:
            cache.preallocate(1,length,torch.bfloat16,torch.device('cuda'))
        return caches

    def logits_for(ids, caches):
        if baseline:
            hidden = model.drop(model.tok_emb(ids))
            for block,cache in zip(model.blocks,caches):
                hidden = block(hidden,cache=cache)
            hidden = model.ln_final(hidden[:,-1:])
        else:
            hidden = _trunk_cached(model,ids,caches)[:,-1:]
        return _lm_head_fp32(model,hidden,4096)[:,0]

    def state_bytes(caches):
        seen=set()
        def visit(value):
            if torch.is_tensor(value):
                storage=value.untyped_storage()
                key=(value.device,storage.data_ptr())
                if key not in seen:
                    seen.add(key)
                    return storage.nbytes()
            elif isinstance(value,dict):
                return sum(visit(v) for v in value.values())
            elif isinstance(value,(tuple,list)):
                return sum(visit(v) for v in value)
            return 0
        return sum(visit({k:v for k,v in c.__dict__.items() if k!='attn'}) for c in caches)

    generator = torch.Generator().manual_seed(20261108)
    tiny=torch.randint(0,32000,(1,257),generator=generator).cuda()
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        caches=caches_for(257,9)
        current=tiny
        output=[]
        for _ in range(9):
            current=logits_for(current,caches).argmax(-1)[:,None]
            output.append(current)
        result=torch.cat(output,1)
        reference=(baseline_cached_generate(model,tiny,9,dtype=torch.bfloat16) if baseline else
                   cached_generate(model,tiny,9,dtype=torch.bfloat16,linear=False))[:,257:]
        assert torch.equal(result,reference), 'Benchmark path differs from the established generator'
    put(args.output/'parity.json',dict(status='passed',prompt_tokens=257,generated_tokens=9,
        token_ids=result.cpu().tolist(),reference='established cached generator'))
    del caches,current,output,result,reference,tiny
    manifest=dict(arm=args.arm,checkpoint=str(args.checkpoint),checkpoint_sha256=args.checkpoint_sha256,
        source_commit=(args.source/'SOURCE_COMMIT').read_text().strip(),
        source_manifest_sha256=sha(args.source/'SOURCE_MANIFEST.json'),helper_sha256=sha(__file__),
        torch=torch.__version__,cuda=torch.version.cuda,hardware=torch.cuda.get_device_name(0),gpu_uuid=gpu_uuid,
        container_pid=os.getpid(),nvml_compute_pid=owner_pid,
        geometry=config,attn_type=attn_type,attention_kwargs=kwargs,weight_dtype='bfloat16',
        positional_buffers='float32',autocast='bfloat16',decode_mode='ordinary cached; fast attention kernels enabled',
        environment={k:v for k,v in os.environ.items() if k.startswith(('FS_','FSL_','KV_','CUDA_VISIBLE'))},
        steps=args.steps,repeats=args.repeats,warmup=1,prefill_output='last-position logits',
        decode_input='fixed random token sequence shared across models',seed=20261108,
        measured_scope='cache allocation and full cached prefill; 257 successive decode steps and output heads',
        timing='synchronized wall time including host overhead; median of three retained repetitions',started_at=time.time())
    put(args.output/'manifest.json',manifest)
    results=[]
    for length in args.lengths:
        generator=torch.Generator().manual_seed(20261108+length)
        all_ids=torch.randint(0,32000,(1,length+args.steps),generator=generator).cuda()
        prompt=all_ids[:,:length]
        observations=[]
        for repeat in range(args.repeats+1):
            exclusive()
            gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
                start=time.perf_counter()
                caches=caches_for(length,args.steps)
                logits=logits_for(prompt,caches)
                torch.cuda.synchronize()
                prefill_seconds=time.perf_counter()-start
                assert torch.isfinite(logits).all()
                prefill_peak=torch.cuda.max_memory_allocated()
                before=state_bytes(caches)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize();start=time.perf_counter()
                for step in range(args.steps):
                    logits=logits_for(all_ids[:,length+step:length+step+1],caches)
                torch.cuda.synchronize()
                decode_seconds=time.perf_counter()-start
                assert torch.isfinite(logits).all()
                decode_peak=torch.cuda.max_memory_allocated()
                after=state_bytes(caches)
            row=dict(repeat=repeat,warmup=repeat==0,prefill_seconds=prefill_seconds,
                decode_seconds=decode_seconds,prefill_peak_bytes=prefill_peak,decode_peak_bytes=decode_peak,
                attention_state_before_bytes=before,attention_state_after_bytes=after)
            observations.append(row)
            put(args.output/'progress.json',dict(status='running',length=length,repeat=repeat,
                completed_lengths=[x['length'] for x in results],current=observations))
            del logits,caches
        measured=observations[1:]
        pre=statistics.median(x['prefill_seconds'] for x in measured)
        dec=statistics.median(x['decode_seconds'] for x in measured)
        results.append(dict(length=length,prefill_seconds=pre,decode_seconds=dec,
            prefill_ktokens_per_second=length/pre/1000,decode_tokens_per_second=args.steps/dec,
            observations=observations))
        put(args.output/'measurements.json',dict(status='running',results=results))
        del all_ids,prompt
    put(args.output/'result.json',dict(status='complete',results=results,
        manifest_sha256=sha(args.output/'manifest.json'),finished_at=time.time()))
    put(args.output/'progress.json',dict(status='complete',lengths=args.lengths,finished_at=time.time()))


if __name__=='__main__':
    main()
