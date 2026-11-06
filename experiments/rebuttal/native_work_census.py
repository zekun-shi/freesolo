"""Capture native kernel shapes, routing counts and maintenance across decoding.

The census is evidence for the FLOP derivation, not a completed FLOP total or
a speed benchmark. FS-B uses the separately validated unfused decode chain so
its beam visits and candidate counts remain observable.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import functools
import gc
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments/scaling')]

import torch
import bench_efficiency as bench
from decode_graph import GraphedDecoder,FSBGraphedDecoder
from src.utils import trunk_triton as tt


def metadata(value):
    if torch.is_tensor(value):
        return dict(shape=list(value.shape),dtype=str(value.dtype))
    if isinstance(value,(int,float,str,bool)) or value is None:return value
    if isinstance(value,dict):
        return {k:metadata(v) for k,v in value.items() if torch.is_tensor(v)
            or isinstance(v,(int,float,str,bool)) or k=='lean'}
    if isinstance(value,(tuple,list)) and len(value)<20:return [metadata(x) for x in value]
    return type(value).__name__


class Census:
    FUNCTIONS={
        'src.fsattn.fsl_select_cuda':['fsl_fast_select','fsl_dense_fast_select','cell_stats',
            'cache_cell_stats','prefix_topp','fsl_dense_decode_topk'],
        'src.fsattn.fsb_fast_step':['forest_beam_fused','forest_beam_pair','union_rerank_fast',
            'rerank_top_fast','decode_readout_fast'],
        'src.fsattn.fsb_rows':['union_rerank_rows'],
        'src.fsattn.fsb_build_cuda':['build_forests_cuda','lean_dirty_build','lean_dirty_build_many'],
        'src.fsattn.cell_select_triton':['indexed_cell_attention'],
        'src.fsattn.readout_triton':['readout_combine'],
        'src.utils.trunk_triton':['sr_decode_fused','sr_decode_fused_int8','fsl_pool_refresh',
            'fsl_lr_decode_combine','lr_comb_int8','lr_decode_read','lr_combine','lr_scale_combine'],
    }

    def __init__(self):
        self.calls=Counter();self.work=Counter();self.patches=[]

    def __enter__(self):
        for path,names in self.FUNCTIONS.items():
            module=importlib.import_module(path)
            for name in names:
                original=getattr(module,name)
                self.patches.append((module,name,original))
                setattr(module,name,self.wrap(path+'.'+name,original))
        return self

    def __exit__(self,*exc):
        for module,name,original in reversed(self.patches):setattr(module,name,original)

    def wrap(self,name,original):
        signature=inspect.signature(original)
        @functools.wraps(original)
        def run(*args,**kwargs):
            values=signature.bind(*args,**kwargs).arguments
            key=json.dumps(dict(function=name,inputs={k:metadata(v) for k,v in values.items()
                if k not in ['out','pv_out','scratch','stats_out','ovf_out']}),sort_keys=True)
            self.calls[key]+=1
            result=original(*args,**kwargs)
            short=name.rsplit('.',1)[-1]
            if result is None:return result
            if short in ['forest_beam_fused','forest_beam_pair']:
                beams=[result] if short=='forest_beam_fused' else result
                for cells,visits in beams:
                    self.work['beam_node_visits']+=int(visits.sum().item())
                    self.work['beam_candidate_slots']+=cells.numel()
                    self.work['beam_valid_candidates']+=int((cells>=0).sum().item())
            if short in ['union_rerank_rows','union_rerank_fast']:
                a,b=values['sel_a'],values['sel_b']
                merged=a if b is None else torch.cat([a,b],-1)
                ordered=merged.sort(-1).values
                unique=(ordered>=0)
                unique[...,1:] &= ordered[...,1:]!=ordered[...,:-1]
                self.work['rerank_query_unique_cells']+=int(unique.sum().item())
                if short=='union_rerank_rows':
                    row=merged.flatten(1).sort(-1).values
                    valid=row>=0;valid[:,1:] &= row[:,1:]!=row[:,:-1]
                    self.work['rerank_block_union_cells']+=int(valid.sum().item())
                    self.work['rerank_block_query_cell_pairs']+=int(valid.sum().item())*a.shape[1]
                cells=result[0]
                self.work['rerank_read_cells']+=int((cells>=0).sum().item())
            return result
        return run

    def report(self):
        return dict(kernel_calls=[dict(json.loads(k),calls=v) for k,v in self.calls.items()],
            observed_work=dict(self.work),complete_flop_accounting=False)


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def attention_state_bytes(decoder):
    objects=set();storages=set();total=0
    def visit(value):
        nonlocal total
        if id(value) in objects:return
        objects.add(id(value))
        if torch.is_tensor(value):
            storage=value.untyped_storage();key=(value.device,storage.data_ptr())
            if key not in storages:
                storages.add(key);total+=storage.nbytes()
        elif isinstance(value,dict):
            for name,item in value.items():
                if name not in ['attn','model','m']:visit(item)
        elif isinstance(value,(tuple,list)):
            for item in value:visit(item)
        elif hasattr(value,'__dict__') and not isinstance(value,torch.nn.Module):
            visit(vars(value))
    visit(decoder.cache)
    visit(getattr(decoder,'layer_bufs',getattr(decoder,'buf',None)))
    return total


@torch.no_grad()
def drive(model,ids,serve,steps,census):
    d=(FSBGraphedDecoder(model,ids,steps+8) if serve=='fsb'
       else GraphedDecoder(model,ids,ids.numel()+steps+8))
    with census() as c:
        d.prefill()
    prefill=c.report() if c else None
    routing=[]
    if serve=='fsb':
        for a in d.attn:
            st=a._fsb_stats
            routing.append({key:float(st[key].sum().item()) for key in ['n_read','n_visit']})
    initial=dict(next_token_sha256=tensor_sha(d.next_tok),persistent_bytes=attention_state_bytes(d),routing=routing)
    teacher=torch.randint(0,32000,(steps,),generator=torch.Generator().manual_seed(4321+ids.numel())).cuda()
    with census() as c:
        for step in range(steps):
            d.next_tok=teacher[step:step+1]
            d.tok_static.copy_(d.next_tok.view(1,1))
            if serve=='fsb':
                d.pos_t.fill_(d.pos)
                d._eager_step()
                logits=d.static_logits
            else:
                d._refresh_window()
                if d.pos % d.Bq==0:d._refresh_block_pool()
                tt._GRAPH_WARMUP=True
                for block in d.blocks:block.ffn._use_gu_cache=True
                try:
                    with torch.autocast('cuda',dtype=torch.bfloat16):logits=d._trunk_graph()
                finally:
                    tt._GRAPH_WARMUP=False
                    for block in d.blocks:block.ffn._use_gu_cache=False
                d._advance_cache_bookkeeping();d.pos+=1
    decode=c.report() if c else None
    assert d.pos==ids.numel()+steps
    final=dict(logits_sha256=tensor_sha(logits),persistent_bytes=attention_state_bytes(d),steps=steps)
    kept=logits.clone()
    del d;gc.collect();torch.cuda.empty_cache()
    return dict(prefill=prefill,decode=decode,initial=initial,final=final),kept


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--serve',choices=['dense','sparse','fsb'],required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--lengths',nargs='+',type=int,default=[4096,8192,16384,32768,65536,131072])
    p.add_argument('--steps',type=int,default=257)
    p.add_argument('--check-parity',action='store_true')
    p.add_argument('--union-calibration',default='')
    a=p.parse_args()
    assert a.steps>=256
    os.environ.update(FS_STRICT_CAUSAL='1',FS_FUSED_SELECT='1',FSL_LINEAR_CACHE='0',
        FSB_PREFILL_STATS='1',FSB_DECODE_V2='0',FSB_TRAIN_GRAPH='0',FSL_SLIDING_MODE='chunked')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_cudnn_sdp(False)
    with Path(a.output).open('x') as f:
        def emit(row):f.write(json.dumps(row)+'\n');f.flush();print(row['kind'],row.get('length',''),flush=True)
        emit(dict(kind='manifest',args=vars(a),source=(ROOT/'SOURCE_COMMIT').read_text().strip(),
            checkpoint_sha256=bench._file_sha256(Path(a.checkpoint)/'model.safetensors'),
            torch=torch.__version__,gpu=torch.cuda.get_device_name(),
            attn_flags=bench.attn_flags(),fsb_flags=bench.fsb_flags(),
            scope='Native shape/routing census; no total FLOP or timing claim'))
        for length in a.lengths:
            model=bench.load(length,a.serve,a.checkpoint,.5 if a.serve=='fsb' else 0)
            if a.serve=='fsb':
                assert a.union_calibration
                options=argparse.Namespace(fsb_union_tau_quantile=.5,fsb_union_tau_file=a.union_calibration,checkpoint=a.checkpoint)
                bench.apply_fsb_tau(model,options,a.output)
            ids=torch.randint(0,32000,(1,length),generator=torch.Generator().manual_seed(1234+length)).cuda()
            reference=None
            if a.check_parity:reference,ref_logits=drive(model,ids,a.serve,a.steps,contextlib.nullcontext)
            result,logits=drive(model,ids,a.serve,a.steps,Census)
            result.update(kind='census',length=length,parameters=model.count_params(),prompt_sha256=tensor_sha(ids))
            if reference is not None:
                result['parity']=dict(initial_token_equal=reference['initial']['next_token_sha256']==result['initial']['next_token_sha256'],
                    final_logits_equal=torch.equal(logits,ref_logits),max_abs=float((logits-ref_logits).abs().max()))
            result['fast_status']=dict(fsl=bench.fsl_fast_status(),fsb=bench.fsb_fast_status())
            emit(result)
            if reference is not None:assert all(result['parity'][k] for k in ['initial_token_equal','final_logits_equal']),result['parity']
            del model,logits,ids;gc.collect();torch.cuda.empty_cache()
        emit(dict(kind='completion',status='complete'))


if __name__=='__main__':main()
