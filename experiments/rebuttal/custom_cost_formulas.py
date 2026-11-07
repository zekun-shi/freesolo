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
