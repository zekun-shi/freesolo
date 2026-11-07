"""Combine disjoint native arithmetic records, retaining incomplete components.

This module deliberately refuses to label a partial arithmetic sum a model
total. The remaining selection/forest formulas can be added without rerunning
the already retained tensor-shape and routing census.
"""
from collections import Counter
import math

from experiments.rebuttal import custom_cost_formulas as costs


def aggregate_common_kernels(phase, *, read_top=4):
    components = {}
    pending = Counter()
    accounted_calls = Counter()
    wrapper_calls = Counter()
    calls = phase['kernel_calls']
    cell_stats_calls = sum(c['calls'] for c in calls if c['function'].endswith('.cell_stats'))

    def add(name, amount, repeats=1):
        components.setdefault(name, Counter()).update({k:v*repeats for k,v in amount.items()})

    readout_rows = 0
    readout_geometry = set()
    for call in calls:
        name = call['function'].rsplit('.',1)[-1]
        args, repeats = call['inputs'], call['calls']
        if name == 'cell_stats':
            b,h,_,d = args['K']['shape']
            assert args['s']==32 and d==64
            add('cell_statistics', costs.cell_statistics(b*h*args['M'],args['s'],d), repeats)
        elif name == 'cache_cell_stats':
            # This wrapper calls the observed cell_stats. Its tensor casts and
            # layout changes add no arithmetic; never count the summaries twice.
            wrapper_calls[name] += repeats
            continue
        elif name == 'quantize_cells_fast':
            b,h,n,d = args['x']['shape']
            assert n % args['s']==0
            add('quantization',costs.quantize_cells(b*h*n//args['s'],args['s'],d),repeats)
        elif name == 'readout_combine':
            b,h,n,d=args['Q']['shape']
            assert args['sel']['shape'][-1]==read_top
            readout_rows += b*h*n*repeats
            readout_geometry.add((args['cell_size'],d))
        elif name == 'sr_decode_fused_int8':
            b,h,n,d=args['Q']['shape']
            assert n==1 and args['use_rope'] and args['fs_uniform_sr']
            window=args['win_idx']['shape'][0]
            assert window==args['window_size']+1
            add('local_decode',costs.short_decode_int8(b*h,window,d),repeats)
        elif name == 'lr_decode_read':
            # Outer wrapper: all arithmetic belongs to lr_comb_int8 below.
            assert args['st'] is not None
            wrapper_calls[name] += repeats
            continue
        elif name == 'lr_comb_int8':
            b,h,n,d=args['q_t']['shape']
            assert n==1 and args['out_short'] is None and not args['pool']['empty']
            add('long_decode',costs.streaming_decode_readout_int8(
                b*h,read_top,args['st'][-1],d),repeats)
            # Dense FS has an independently observed selection kernel. FS-L
            # performs candidate scoring and top-K inside this function.
            if not args['pool'].get('dense',False):
                pending['lr_comb_int8.selection'] += repeats
        elif name == 'decode_readout_fast':
            b,h,n,d=args['q_t']['shape']
            assert n==1 and args['sr'] is None
            assert args['sel']['shape'][-1]==read_top
            add('long_decode',costs.flat_decode_readout_int8(b*h,read_top,32,d),repeats)
        elif name == 'lr_combine':
            add('branch_combine',dict(flops=math.prod(args['out_short']['shape'])),repeats)
        elif name == 'lr_scale_combine':
            add('branch_combine',dict(flops=2*math.prod(args['out_short']['shape'])),repeats)
        else:
            pending[name] += repeats
            continue
        accounted_calls[name] += repeats

    assert cell_stats_calls >= wrapper_calls['cache_cell_stats']
    assert accounted_calls['lr_comb_int8']==wrapper_calls['lr_decode_read']
    if readout_rows:
        assert len(readout_geometry)==1
        size,width=next(iter(readout_geometry))
        work=phase['observed_work']
        assert work['readout_tile_token_pairs']==work['readout_group_union_cells']*64*size
        add('long_prefill',costs.fused_prefill_readout(
            work['readout_group_union_cells'],readout_rows,64,size,width))
    total=Counter()
    for counts in components.values():total.update(counts)
    return dict(covered_kernel_arithmetic=dict(total),
        components={k:dict(v) for k,v in components.items()},
        accounted_calls=dict(accounted_calls),wrappers_not_double_counted=dict(wrapper_calls),
        pending_kernel_components=dict(pending),complete_kernel_coverage=not pending)


def aggregate_row(row):
    """Preserve the full ATen ledger and add only independently derived kernels."""
    result=dict(length=row['length'],steps=row['final']['steps'],phases={})
    for name in ['prefill','decode']:
        phase=row[name]
        aten=phase['aten_outside_observed_kernels']
        assert aten['complete_operator_coverage'] and not aten['unknown_operators']
        kernels=aggregate_common_kernels(phase)
        partial=Counter(aten['total']);partial.update(kernels['covered_kernel_arithmetic'])
        result['phases'][name]=dict(aten=aten,opaque_kernels=kernels,
            accounted_arithmetic=dict(partial),
            complete_flop_accounting=kernels['complete_kernel_coverage'])
    return result
