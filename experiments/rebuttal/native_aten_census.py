"""Count native PyTorch work outside opaque kernels and census those kernels.

The two ledgers are disjoint: every observed native kernel runs with the ATen
ledger suspended, including its observer. Native kernel work still needs its
separate arithmetic derivation. These measurements are never timed.
"""
from __future__ import annotations

import functools
import inspect
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments/scaling')]

from experiments.rebuttal import native_work_census as native
from experiments.rebuttal.operation_ledger import OperationLedger

BaseCensus=native.Census
active_model=None


class NativeAtenCensus(BaseCensus):
    FUNCTIONS=dict(BaseCensus.FUNCTIONS,**{
        'src.fsattn.kv_int8':['quantize_cells_fast'],
        'src.fsattn.fsb_build_fast':['build_spines_fast'],
        'src.fsattn.fsb_beam_cuda':['spines_tables_cuda'],
    })

    def __init__(self):
        super().__init__()
        self.ledger=OperationLedger()
        self.method_patches=[]
        self.spines=[]
        self.slot_work=[]

    def __enter__(self):
        from src.fsattn.kv_bounded import SlotStore
        self.ledger.install_bounded_kernels(include_readout=False)
        self.ledger.attach_model(active_model)
        self.ledger._patch(native.tt,'rope',lambda x,cos,sin,out_dtype=None:
            self.ledger.add(flops=3*x.numel(),custom_rope_elements=x.numel()))
        for block in active_model.blocks:
            attention=block.attn.fsattn
            for name in ['graph_decode_layer','graph_decode_layer_fsb']:
                original=getattr(attention,name)
                @functools.wraps(original)
                def scoped(*args,_original=original,**kwargs):
                    with self.ledger.in_scope('attention'):
                        return _original(*args,**kwargs)
                self.method_patches.append((attention,name,original))
                setattr(attention,name,scoped)
        for name in ['prefill','store_cell','pool']:
            original=getattr(SlotStore,name)
            signature=inspect.signature(original)
            @functools.wraps(original)
            def slot_call(store,*args,_original=original,_name=name,_signature=signature,**kwargs):
                # Keep the original Python arithmetic in the ATen ledger.
                # Only the opaque kernels below need additional formula work.
                result=_original(store,*args,**kwargs)
                with self.ledger.suspend():
                    values=_signature.bind(store,*args,**kwargs).arguments
                    record=dict(method=_name,heads=store.B*store.H,width=store.D,size=store.s,
                        flags=store.flags.tolist())
                    assert record['flags']==[0,0],record
                    if _name=='prefill':
                        cells=values['N']//store.s
                        record.update(launched_cells=store.B*store.H*cells,
                            retained_cells=int((store.slot_of[:,:,:cells]>=0).sum().item()))
                    elif _name=='store_cell':
                        record.update(launched_cells=store.B*store.H,cell=values['cell'])
                    else:
                        record.update(candidate_slots=0 if result.get('empty',True) else result['C'])
                    self.slot_work.append(record)
                return result
            self.method_patches.append((SlotStore,name,original))
            setattr(SlotStore,name,slot_call)
        self.ledger.__enter__()
        return super().__enter__()

    def __exit__(self,*exc):
        try:
            return super().__exit__(*exc)
        finally:
            self.ledger.__exit__(*exc)
            for obj,name,original in reversed(self.method_patches):
                setattr(obj,name,original)
            self.ledger.close()

    def wrap(self,name,original):
        observed=super().wrap(name,original)
        signature=inspect.signature(original)
        @functools.wraps(original)
        def run(*args,**kwargs):
            # Both custom-kernel Python internals and observer tensor operations
            # are excluded. Their arithmetic is derived from the native census.
            with self.ledger.suspend():
                result=observed(*args,**kwargs)
                short=name.rsplit('.',1)[-1]
                if result is not None and short in ['build_spines_fast','spines_tables_cuda']:
                    values=signature.bind(*args,**kwargs).arguments
                    for key in ['m_b','m_b2']:
                        counts=values.get(key)
                        if counts is None:continue
                        counts=counts.reshape(-1).tolist()
                        self.spines.append(dict(function=short,argument=key,rows=len(counts),
                            nonempty_rows=sum(n>0 for n in counts),
                            merges=sum(max(0,int(n).bit_count()-1) for n in counts),
                            chunk_roots=sum(int(n).bit_count() for n in counts)))
                return result
        return run

    def report(self):
        result=super().report()
        result.update(aten_outside_observed_kernels=self.ledger.report(),spine_builds=self.spines,
            slot_store_kernels=self.slot_work)
        return result


original_drive=native.drive


def drive_with_model(model,*args,**kwargs):
    global active_model
    active_model=model
    try:
        return original_drive(model,*args,**kwargs)
    finally:
        active_model=None


if __name__=='__main__':
    native.Census=NativeAtenCensus
    native.drive=drive_with_model
    native.main()
