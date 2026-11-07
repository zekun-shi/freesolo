"""Logical inference arithmetic for the trained rebuttal attention controls.

Use valid attention pairs, FMA=2 and reduction length minus one. This is
algorithmic work, not a hardware-instruction count: tile padding, repeated
tile loads/rotations, rotary-table construction, allocation and copies are
excluded. Transcendentals and comparisons remain separate. Dense-LR keeps
both branches, including their overlap and the ordinary FS cache statistics.
"""
from collections import Counter
from dataclasses import asdict, dataclass

from experiments.rebuttal.custom_cost_formulas import cell_statistics


@dataclass(frozen=True)
class Geometry:
    hidden: int = 1024
    ffn: int = 2816
    heads: int = 16
    layers: int = 24
    vocabulary: int = 32000
    window: int = 512
    query_block: int = 128
    cell_stride: int = 32

    def __post_init__(self):
        assert min(asdict(self).values()) > 0
        assert self.hidden % self.heads == 0
        assert self.width % 2 == 0
        assert self.query_block % self.cell_stride == 0

    @property
    def width(self):
        return self.hidden // self.heads


def prefix_pairs(length, mode, *, window=512, query_block=128):
    """Pairs over absolute query positions [0, length), per head/layer."""
    assert length >= 0 and window > 0 and query_block > 0
    if mode == 'global':
        return length * (length + 1) // 2
    if mode == 'local':
        cap = window + 1  # W previous tokens and the current token.
        short = min(length, cap)
        return short * (short + 1) // 2 + max(0, length - cap) * cap
    if mode == 'dense_lr':
        blocks, tail = divmod(length, query_block)
        return query_block**2 * blocks * (blocks - 1) // 2 + tail * blocks * query_block
    raise ValueError(mode)


def attention_work(mode, length, *, steps=None, geometry=Geometry()):
    """Prefill [0,N), or decode positions N through N+steps-1."""
    assert length > 0 and (steps is None or steps > 0)
    start, end = (0, length) if steps is None else (length, length + steps)
    args = dict(window=geometry.window, query_block=geometry.query_block)
    pairs = prefix_pairs(end, mode, **args) - prefix_pairs(start, mode, **args)
    rows = end - start
    if mode == 'dense_lr':
        rows = max(0, end - max(start, geometry.query_block))
    pairs *= geometry.heads
    rows *= geometry.heads
    return dict(flops=pairs * (4 * geometry.width + 4) - rows,
                exp=pairs, fp_comparisons=pairs - rows,
                valid_token_pairs=pairs, nonempty_attention_rows=rows,
                qk_pv_flops=4 * geometry.width * pairs)


def parameter_count(arm, geometry=Geometry()):
    assert arm in ('nope', 'swan', 'dense_lr')
    g = geometry
    base = g.layers * (4*g.hidden**2 + 3*g.hidden*g.ffn + 2*g.hidden)
    base += g.hidden*g.vocabulary + g.hidden
    if arm == 'dense_lr':
        # sel_scale and long_range_scale per head; shared head-width LR RMS.
        # Fixed log_sigma is a buffer and is not counted as a parameter.
        base += g.layers * (2*g.heads + g.width)
    return base


def model_cost(arm, length, *, steps=None, geometry=Geometry()):
    """Full logical arithmetic under each control's ordinary cached decoder."""
    assert arm in ('nope', 'swan', 'dense_lr')
    assert length > 0 and (steps is None or steps > 0)
    g = geometry
    n = length if steps is None else steps
    head_rows = 1 if steps is None else steps
    # FS forward_cached normalizes all prompt positions; baseline_cached_generate
    # crops the hidden state before its final RMS normalization.
    norm_rows = n if arm == 'dense_lr' else head_rows
    pieces = dict(
        mlp=dict(flops=g.layers*n*(6*g.hidden*g.ffn + 5*g.ffn), exp=g.layers*n*g.ffn),
        block_norms_residuals=dict(flops=g.layers*n*(10*g.hidden+2), rsqrt=2*g.layers*n),
        final_norm=dict(flops=norm_rows*(4*g.hidden+1), rsqrt=norm_rows),
        output_head=dict(flops=head_rows*2*g.hidden*g.vocabulary),
        projections=dict(flops=g.layers*n*8*g.hidden**2),
    )

    def branch(name, mode, layers):
        if layers:
            pieces[name] = {k: v*layers for k, v in attention_work(
                mode, length, steps=steps, geometry=g).items()}

    if arm == 'nope':
        branch('global_attention', 'global', g.layers)
    elif arm == 'swan':
        global_layers = (g.layers + 3) // 4
        local_layers = g.layers - global_layers
        branch('global_attention', 'global', global_layers)
        branch('local_attention', 'local', local_layers)
        pieces['local_rope_rotation'] = dict(flops=local_layers*n*6*g.hidden)
        # log_a(a+position), then multiply every query channel.
        pieces['global_inference_scaling'] = dict(
            flops=global_layers*n*(g.hidden+2), log=global_layers*n)
    else:
        branch('local_attention', 'local', g.layers)
        branch('long_range_attention', 'dense_lr', g.layers)
        pieces['long_range_norm_and_mix'] = dict(
            flops=g.layers*n*(6*g.hidden+g.heads), rsqrt=g.layers*n*g.heads)
        if steps is None:
            rotations = 2*n
            completed_cells = length // g.cell_stride
        else:
            # The LR cache retains unrotated keys: every local-window key is
            # rotated again during decode. Include the current eager path's
            # initial Q/K rotation in addition to _sr_decode_step's rotations.
            local_pairs = (prefix_pairs(length+steps, 'local', window=g.window)
                           - prefix_pairs(length, 'local', window=g.window))
            rotations = 3*n + local_pairs
            completed_cells = (length+steps)//g.cell_stride - length//g.cell_stride
            pieces['local_decode_mask'] = dict(
                flops=g.layers*local_pairs*(g.heads+2),
                fp_comparisons=g.layers*local_pairs)
        pieces['local_rope_rotation'] = dict(flops=g.layers*rotations*3*g.hidden)
        pieces['cache_cell_statistics'] = cell_statistics(
            g.layers*g.heads*completed_cells, g.cell_stride, g.width)
        pieces['fixed_sigma_materialization'] = dict(exp=g.layers*(1 if steps is None else steps)*g.heads)

    total, attention = Counter(), Counter()
    outside = {'mlp', 'block_norms_residuals', 'final_norm', 'output_head'}
    for name, counts in pieces.items():
        total.update(counts)
        if name not in outside:
            attention.update(counts)
    return dict(arm=arm, length=length, steps=steps, geometry=asdict(g),
                parameters=parameter_count(arm, g), components=pieces,
                total=dict(total), attention=dict(attention),
                prefill_head_positions=head_rows,
                scope='Logical arithmetic; valid attention pairs; ordinary cached control decoder')
