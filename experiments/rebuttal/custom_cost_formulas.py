"""Arithmetic for specific opaque kernels, to combine with disjoint ATen counts.

FMA costs two operations. Reductions use n-1 additions. Transcendentals,
rounding and comparisons are separate. These functions do not constitute a
whole-model total; selection, forest construction and other kernels must also
be accounted for. Source references identify the operations being counted.
"""


def cell_statistics(cells, size, width):
    """fsl_select_cuda.cell_stats: mean keys and maximum token norm per cell."""
    assert cells >= 0 and size > 0 and width > 0
    # Mean: size*width. Token squared norms: size*(2*width-1).
    return dict(flops=cells*size*(3*width-1), sqrt=cells*size,
                fp_comparisons=cells*(size-1))


def quantize_cells(cells, size, width):
    """kv_int8._quant_cells_kernel, including abs, scale and element division."""
    assert cells >= 0 and size > 0 and width > 0
    elements=cells*size*width
    return dict(flops=2*elements+cells, round=elements,
                fp_comparisons=cells*size*width)


def fused_prefill_readout(union_cells, query_rows, group, size, width):
    """readout_triton._readout_combine_kernel using observed group unions.

Each union cell incurs QK and PV products, scale/subtract/reduction work,
online softmax rescaling and accumulator rescaling for every query in its
group. Final division, RMS normalization and branch combination cover all
query rows. union_cells sums the distinct cells across groups and heads;
query_rows includes all heads/layers. Integer union/sorting work is separate.
"""
    assert union_cells >= 0 and query_rows >= 0 and min(group,size,width) > 0
    updates=union_cells*group
    # Per update: 4*size*width GEMM operations; 3*size+2 softmax arithmetic;
    # 2*width accumulator rescaling/combining. Final row: 7*width+1.
    return dict(flops=updates*(4*size*width+3*size+2*width+2)
                      +query_rows*(7*width+1),
                exp=updates*(size+1),rsqrt=query_rows,
                fp_comparisons=updates*(size+1)+query_rows)


def short_decode_int8(rows, window, width):
    """Uniform local branch: dequantization, RoPE, two GEMMs and softmax.

    Matches trunk_triton.sr_decode_fused_int8 with use_rope=True and
    fs_uniform_sr=True. window includes the current token. Nominal logical
    elements are counted, including the dequantization expression evaluated
    before selecting hot rows. Integer indexing and copies are separate.
    """
    assert rows >= 0 and window > 0 and width > 0 and width % 2 == 0
    return dict(flops=rows*(9*window*width+3*width+6*window-1),
                exp=rows*window, fp_comparisons=rows*(2*window-1))


def streaming_decode_readout_int8(rows, cells, size, width):
    """FS/FS-L selected-cell readout, excluding selection and SR addition.

    The _lrc/_fsd_comb_int8 kernels use explicit multiply/reduce expressions,
    unlike the prefill tensor-core GEMMs. Count each such reduction with n-1
    additions. Includes online softmax, dequantization, RMS and long scaling.
    All selected cells must be valid and distinct; rows with empty reads need
    a separate census. Inputs in the rebuttal grid are beyond the cold prefix.
    """
    assert rows >= 0 and min(cells,size,width) > 0
    return dict(flops=rows*(cells*(6*size*width+2*size+width+2)+6*width+1),
                exp=rows*cells*(size+1),rsqrt=rows,
                fp_comparisons=rows*(cells*size+1))


def flat_decode_readout_int8(rows, cells, size, width):
    """FS-B _fsb_decread_int8_kernel with sr=None; no long scaling yet.

    Softmax spans all selected tokens together. The separately observed
    lr_scale_combine supplies the long scale and SR addition (2*rows*width).
    """
    assert rows >= 0 and min(cells,size,width) > 0
    tokens=cells*size
    return dict(flops=rows*(6*tokens*width+3*tokens+3*width),
                exp=rows*tokens,rsqrt=rows,
                fp_comparisons=rows*(tokens-1))


def forest_node(rows, width):
    """Logical lean16 node algebra, without padding or redundant warp copies.

    Includes the mean, bf16 rounding slack, radius, farthest-vector start,
    two power iterations, and final split projection. Dot products expressed
    as multiply/reduce use n-1 additions. The signed-zero padding machinery
    and integer split permutation are not floating arithmetic in this model.
    A leaf stores its mean and rounding slack but performs no split.
    """
    assert rows > 0 and rows & (rows-1)==0 and width > 0
    if rows==1:
        return dict(flops=4*width,sqrt=1,fp_comparisons=1,sort_input_items=0)
    return dict(flops=14*rows*width-4*rows+8*width-2,
                sqrt=5,fp_comparisons=3*rows+1,sort_input_items=rows)


def spine_merge(width):
    """Weighted mean and two enclosing-radius distances for one spine merge."""
    assert width>0
    return dict(flops=10*width+1,sqrt=2,fp_comparisons=2)


def pooled_decode_selection(rows, candidates, width, topk):
    """_lrc_comb_int8_kernel's BF16 candidate scores before selected readout.

    Count the nominal candidate shape; the Triton program evaluates the
    expressions for its padded power-of-two vector before masking. Ceilings
    and centers are supplied by the separately counted slot-pool kernel.
    The denominator is shared by the candidate vector. Selection workloads
    are reported as input items/passes, separately from floating arithmetic.
    """
    assert rows>=0 and candidates>0 and width==64 and 0<topk<=candidates
    padded=1<<(candidates-1).bit_length()
    return dict(flops=rows*(padded*(2*width+7)+1),erf=rows*padded,
                topk_input_items=rows*padded,topk_passes=rows*topk)


def beam_node_scores(visits, queries, width):
    """Logical bound scores of visited nodes, including query norm/denominator.

    Count each actual visited child once. Dots use explicit multiply/reduce;
    redundant warp copies and tensor-core union padding are not part of this
    logical graph. The radius coefficient multiplies each query norm once.
    """
    assert visits>=0 and queries>=0 and width>0
    return dict(flops=visits*(2*width+11)+queries*(2*width+1),
                sqrt=queries,erf=visits,beam_scored_nodes=visits)


def candidate_lse_scores(pairs, size, width, *, int8=False):
    """Canonical candidate LSE+position score, before union/top-K.

    Prefill GEMMs use FMA=2; decode uses multiply/reduce and adds int8
    dequantization. Position uses the complete erf expression for each pair;
    callers must label this as logical arithmetic, not the CUDA far shortcut
    or a hardware-instruction count. Duplicate decode candidates are scored
    independently before the later union deduplicates them.
    """
    assert pairs>=0 and size>0 and width>0
    per_pair=(3*size*width+2*size if int8 else 2*size*width+3*size)+9
    return dict(flops=pairs*per_pair,exp=pairs*size,log=pairs,erf=pairs,
                fp_comparisons=pairs*(size-1))


def dense_prefill_selection(heads, blocks, query_block, size, width, *, sigma=4.0, tile=64):
    """Dense CUDA selection: padded GEMMs and the executed far/near expressions.

    Scalar setup is counted once per logical block/cell, without redundant
    copies of a shared scalar in CUDA threads. Abs counts one operation.
    The first query block exits before arithmetic. Top-K comparison work is
    represented by the number of inserted scores, not an instruction count.
    """
    import struct
    assert heads>0 and blocks>0 and query_block%size==0 and width>0
    threshold=struct.unpack('!f',struct.pack('!I',0x407AD910))[0]
    margin=threshold*2*sigma
    flops=erfs=items=gemm_pairs=0
    for block in range(1,blocks):
        visible=block*query_block//size
        flops+=3+query_block  # block start, denominator, margin, query positions
        for start in range(0,visible,tile):
            count=min(tile,visible-start)
            pairs=query_block*tile
            flops+=pairs*(2*width+2)  # QK, scale, initial score addition
            gemm_pairs+=pairs
            items+=pairs
            if count==tile:
                flops+=3  # last cell center and distance used by all_far
            far=(count==tile and (visible-start-tile+0.5)*size>=margin)
            if far:
                flops+=count  # lambda * ceiling
            else:
                flops+=4*count  # center, distance and lambda * ceiling
                near=sum((visible-cell-0.5)*size<margin for cell in range(start,start+count))
                flops+=5*query_block*near  # abs distance, divide, multiply/add
                erfs+=query_block*near
                items+=query_block*near
    return dict(flops=heads*flops,erf=heads*erfs,
                selection_score_items=heads*items,selection_gemm_pairs=heads*gemm_pairs)


def pooled_prefill_selection(heads, blocks, query_block, size, width, pool, recent, *, sigma=4.0):
    """Fused FS-L selection with all C=pool+recent GEMM candidate slots.

    The recent window contains every near candidate; deduplication retains
    exactly one copy. The top-P kernel uses integer radix keys and adds no
    floating arithmetic. One-time shape/stream cache construction is outside
    this steady-state kernel model.
    """
    import struct
    assert heads>0 and blocks>0 and query_block%size==0 and width>0 and pool>32 and recent>0
    margin=struct.unpack('!f',struct.pack('!I',0x407AD910))[0]*2*sigma
    assert margin<=(recent+0.5)*size
    candidates=pool+recent
    pairs=blocks*query_block*candidates
    flops=pairs*(2*width+2)
    erfs=0
    for block in range(blocks):
        visible=block*query_block//size
        # Top-P pads with nonnegative indices. Negative recent candidates
        # short-circuit the floating visibility expression.
        flops+=2+query_block+5*candidates+2*(pool+min(recent,visible))
        near=sum((visible-cell-0.5)*size<margin for cell in range(max(0,visible-recent),visible))
        flops+=5*query_block*near
        erfs+=query_block*near
    return dict(flops=heads*flops,erf=heads*erfs,
                selection_score_items=heads*(pairs+erfs),selection_gemm_pairs=heads*pairs)


def dense_decode_selection(heads, length, steps, query_block, size, width, splits, *, sigma=4.0):
    """CUDA split top-K: actual visible cells at each autoregressive position."""
    import struct
    assert heads>0 and length>=query_block and steps>0 and query_block%size==0 and width%4==0
    margin=struct.unpack('!f',struct.pack('!I',0x407AD910))[0]*2*sigma
    cells=near=0
    for position in range(length,length+steps):
        visible=(position//query_block)*query_block//size
        cells+=visible
        # Only cells within the distance threshold need a division/erf.
        low=max(0,visible-int(margin//size)-2)
        near+=sum(position-(c+0.5)*size<margin for c in range(low,visible))
    return dict(flops=heads*(cells*(2*width+8)+near+2*splits*steps),
                erf=heads*near,selection_score_items=heads*cells)
