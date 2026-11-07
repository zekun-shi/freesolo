"""Combine disjoint native arithmetic records, retaining incomplete components.

This module deliberately refuses to label a partial arithmetic sum a model
total. The remaining selection/forest formulas can be added without rerunning
the already retained tensor-shape and routing census.
"""
from collections import Counter
import math

from experiments.rebuttal import custom_cost_formulas as costs


def forest_node_population(record):
    """Count valid logical nodes from the captured CUDA build task records.

    Full builds follow spine_chunks; dirty builds follow batch_full and the
    tall proposal filter. Empty/padded chunks write sentinels and contribute
    no logical tree nodes. Rebuilds are counted again when executed again.
    """
    population=Counter()
    admitted=record['admitted']

    def tree(height, copies=1):
        for level in range(height+1):population[1<<level] += copies*(1<<(height-level))

    def tasks(records):
        for kind,forest,head,height,chunk,count,depth,node,flag,dep in records:
            assert forest==0 and 0<=head<len(admitted)
            if kind==0:
                for h in range(head,head+(max(node,1) if height<=5 else 1)):
                    for i in range(count):
                        if (chunk+(i<<depth)+1)*(1<<height)<=admitted[h]:tree(height)
            elif kind in [1,2]:
                if (chunk+1)*(1<<height)>admitted[head]:continue
                if kind==1:
                    assert 0<=depth<height and 0<=node<(1<<depth)
                    population[1<<(height-depth)] += 1
                else:
                    assert depth==height-9 and 0<=node<(1<<depth)
                    tree(9)
            else:raise AssertionError(('unrecognized forest task',kind))

    if record['kind']=='build_forests_cuda':
        selected=record['selected_chunks']
        if selected is not None:
            for height,heads in selected:
                assert len(heads)==len(admitted)
                for count,chunk in zip(admitted,heads):
                    if chunk is not None and (chunk+1)*(1<<height)<=count:tree(height)
        else:
            trim=record['prefill_trim']
            for count in admitted:
                limit=min(count,record['capacity'])
                if trim is not None:limit=min(limit,trim)
                for height in range(record['capacity'].bit_length()):
                    chunks=limit>>height
                    if trim is not None:chunks=(chunks+1)//2
                    tree(height,chunks)
    else:
        assert record['kind'] in ['lean_dirty_build','lean_dirty_build_many']
        tasks(record['tasks'])
        tall=record.get('tall')
        if tall is not None:
            proposals,nkept,_=tall
            old=[n-added for n,added in zip(admitted,record['new_admitted'])]
            def held(n,k,j):return bool((n>>k)&1) and ((n>>(k+1))<<1)==j
            def dirty(k):
                now=any((n>>k)&1 for n in admitted)
                was=any((n>>k)&1 for n in old)
                changed=any(((a>>k)&1)!=((b>>k)&1) or
                    (((a>>(k+1))<<1) if (a>>k)&1 else 0)!=
                    (((b>>(k+1))<<1) if (b>>k)&1 else 0)
                    for a,b in zip(admitted,old))
                return now and (not was or changed)
            kept=[]
            for index,task in enumerate(proposals):
                _,_,head,k,j,*_=task
                now,was=held(admitted[head],k,j),held(old[head],k,j)
                keep=(now and not was) if index<len(proposals)-nkept else (dirty(k) and now and was)
                if keep:
                    kept.append(task)
            tasks(kept)
    return {size:n for size,n in sorted(population.items()) if n}


def forest_algebra(phase):
    """Logical node/spine algebra; admission and layout overhead remain separate."""
    total=Counter()
    nodes=Counter()
    internal_spine_merges=0
    for record in phase['forest_builds']:
        population=forest_node_population(record)
        nodes.update(population)
        for size,count in population.items():
            total.update({k:v*count for k,v in costs.forest_node(size,record['head_dim']).items()})
        if record['kind'].startswith('lean_dirty_build'):
            merges=sum(max(0,n.bit_count()-1) for n in record['admitted'])
            internal_spine_merges += merges
            total.update({k:v*merges for k,v in costs.spine_merge(record['head_dim']).items()})
    external_spine_merges=sum(s['merges'] for s in phase.get('spine_builds',[]))
    if external_spine_merges:
        assert all(record['head_dim']==64 for record in phase['forest_builds'])
        total.update({k:v*external_spine_merges for k,v in costs.spine_merge(64).items()})
    return dict(logical_algebra=dict(total),nodes_by_population=dict(nodes),
        internal_spine_merges=internal_spine_merges,external_spine_merges=external_spine_merges,
        scope='Valid logical nodes; padding, integer layout and admission checks are separate')


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
    for slot in phase.get('slot_store_kernels',[]):
        assert slot['flags']==[0,0]
        if slot['method'] in ['prefill','store_cell']:
            # _fill_slots_kernel evaluates its quantization expressions even
            # for masked cells, then masks the stores. Count the launched shape.
            add('slot_quantization',costs.quantize_cells(
                2*slot['launched_cells'],slot['size'],slot['width']))
        elif slot['method']=='pool':
            elements=slot['heads']*slot['candidate_slots']
            add('slot_pool',dict(flops=4*elements,fp_comparisons=elements))
        else:raise AssertionError(('unrecognized slot method',slot['method']))
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
