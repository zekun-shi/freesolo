"""Collect bounded FS-L work with the real prefill and graph-step bodies."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments/scaling')]

import torch
from experiments.rebuttal import bounded_benchmark as bench
from experiments.rebuttal.operation_ledger import OperationLedger
from experiments.scaling import linear_decode_graph as linear_graph
from experiments.scaling.longctx_infer import _lm_head_fp32


@contextmanager
def counted(model):
    ledger=OperationLedger()
    ledger.install_bounded_kernels()
    ledger.attach_model(model)
    graph_attention=linear_graph.graph_attention

    def attention(*args,**kwargs):
        with ledger.in_scope('attention'):
            return graph_attention(*args,**kwargs)

    linear_graph.graph_attention=attention
    try:
        with ledger:
            yield ledger
    finally:
        linear_graph.graph_attention=graph_attention
        ledger.close()


def digest_tensor(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@torch.no_grad()
def collect(args):
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    args.dtype=getattr(torch,args.dtype)
    model=bench.build(args)
    manifest=bench.manifest(args)
    manifest['args']=dict(manifest['args'])
    manifest['operation_ledger_sha256']=hashlib.sha256((ROOT/'experiments/rebuttal/operation_ledger.py').read_bytes()).hexdigest()
    manifest['scope']='Shape arithmetic and actual cache maintenance; instrumentation is not timed'
    manifest['args']['dtype']=str(args.dtype)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as sink:
        def emit(row):
            sink.write(json.dumps(row)+'\n');sink.flush()
            print(json.dumps({k:v for k,v in row.items() if k in ['kind','length','status','unknown_operators']}),flush=True)
        emit(dict(kind='manifest',**manifest))
        for length in args.lengths:
            torch.manual_seed(1234+length)
            ids=torch.randint(0,32000,(1,length),device='cuda')
            prompt_sha256=digest_tensor(ids)
            # The accepted graph step is evaluated eagerly so every replay's
            # arithmetic is observable. Its body and maintenance are unchanged.
            decoder=linear_graph.LinearGraphedDecoder(model,ids,length+args.steps+8,dtype=args.dtype)
            with counted(model) as ledger:
                decoder.prefill()
            prefill=ledger.report()
            first=digest_tensor(decoder.next_tok)
            pstate=bench.state_bytes(decoder.cache,decoder.layer_bufs)
            emit(dict(kind='prefill_cost',length=length,prompt_sha256=prompt_sha256,
                next_token_sha256=first,attention_state_bytes=pstate,**prefill))
            if prefill['unknown_operators']:
                raise RuntimeError('Uncounted prefill operators; inventory retained in output')
            teacher=torch.randint(0,32000,(args.steps,),generator=torch.Generator().manual_seed(4321+length))
            boundary_steps=[]
            per_step=[]
            totals=None
            for step in range(args.steps):
                decoder.next_tok=teacher[step:step+1].to('cuda')
                with counted(model) as ledger:
                    decoder.tok_static.copy_(decoder.next_tok.view(1,1))
                    with ledger.in_scope('attention_maintenance'):
                        decoder._refresh_window()
                        if decoder.pos % decoder.Bq == 0:
                            boundary_steps.append(step)
                            decoder._refresh_block_pool()
                    with bench.autocast(args.dtype):
                        logits=decoder._trunk_graph()
                    with ledger.in_scope('attention_maintenance'):
                        decoder._advance_cache_bookkeeping()
                    decoder.pos += 1
                report=ledger.report()
                if report['unknown_operators']:
                    emit(dict(kind='decode_inventory',length=length,step=step,**report))
                    raise RuntimeError('Uncounted decode operators; inventory retained')
                per_step.append(report['total'])
                if totals is None:
                    totals=report
                else:
                    for scope, counts in report['by_scope'].items():
                        target=totals['by_scope'].setdefault(scope,{})
                        for k,v in counts.items():target[k]=target.get(k,0)+v
                    for field in ['total','operator_calls']:
                        for k,v in report[field].items():totals[field][k]=totals[field].get(k,0)+v
            assert decoder.pos==length+args.steps
            assert len(boundary_steps)>=2, boundary_steps
            emit(dict(kind='decode_cost',length=length,steps=args.steps,
                boundary_steps=boundary_steps,per_step_counts=per_step,
                per_token={k:v/args.steps for k,v in totals['total'].items()},
                final_logits_sha256=digest_tensor(logits),
                attention_state_bytes=bench.state_bytes(decoder.cache,decoder.layer_bufs),**totals))
            del decoder,logits
            gc.collect();torch.cuda.empty_cache()
        emit(dict(kind='completion',status='complete',completed_at=time.time()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--config',default='340m')
    p.add_argument('--lengths',nargs='+',type=int,default=[4096,8192,16384,32768,65536,131072,262144])
    p.add_argument('--steps',type=int,default=257)
    p.add_argument('--pool',type=int,default=64)
    p.add_argument('--fused',type=int,default=1)
    p.add_argument('--dtype',default='bfloat16')
    p.add_argument('--weight-dtype',default='bfloat16')
    a=p.parse_args()
    if a.steps<256:p.error('At least 256 steps are required')
    collect(a)


if __name__=='__main__':main()
