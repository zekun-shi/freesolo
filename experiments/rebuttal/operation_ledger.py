"""Shape-based floating-point work, including otherwise opaque FS kernels.

This is an algorithmic operation ledger, not a hardware instruction counter.
Multiply-adds cost two operations. Transcendentals, comparisons, sorting items
and copies are separate quantities. Unknown operators are retained explicitly.
CUDA graph capture/replay must be counted through an equivalent eager body.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from functools import wraps
import math

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)


def matmul_flops(a, b):
    """FMA=2, with broadcast batch dimensions and vector special cases."""
    ash, bsh = tuple(a), tuple(b)
    if len(ash) == 1:
        ash = (1,) + ash
    if len(bsh) == 1:
        bsh = bsh + (1,)
    assert ash[-1] == bsh[-2], (a, b)
    batch = math.prod(torch.broadcast_shapes(ash[:-2], bsh[:-2]))
    return 2 * batch * ash[-2] * ash[-1] * bsh[-1]


def softmax_work(elements, rows):
    return dict(flops=3 * elements - rows, exp=elements,
                fp_comparisons=elements - rows)


class OperationLedger(TorchDispatchMode):
    """Retain every operator family; never silently call a partial sum complete."""
    MOVEMENT = {
        'alias', 'as_strided', 'view', '_unsafe_view', 'reshape', 'permute', 'transpose',
        't', 'squeeze', 'unsqueeze', 'expand', 'slice', 'select', 'split', 'split_with_sizes',
        'unbind', 'detach', 'empty', 'empty_like', 'empty_strided', 'new_empty',
        'zeros', 'zeros_like', 'new_zeros', 'ones', 'ones_like', 'full', 'full_like',
        'scalar_tensor', 'lift_fresh', 'lift_fresh_copy', 'arange', 'clone', 'contiguous',
        'copy_', '_to_copy', 'to', 'cat', 'stack', 'index', 'index_select', 'gather',
        'scatter', 'scatter_', 'index_copy_', 'masked_fill', 'masked_fill_',
        'fill_', 'zero_', 'flip', 'constant_pad_nd', 'embedding', 'repeat',
        'repeat_interleave', 'where', 'tril', 'triu', '_local_scalar_dense',
        'is_same_size', 'record_stream', 'is_pinned', 'resize_', 'set_',
        'promote_types',  # Dtype metadata query; no tensor arithmetic or copy.
    }
    COMPARE = {'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'isfinite', 'isinf', 'isnan',
               'logical_and', 'logical_or', 'logical_not', 'bitwise_and', 'bitwise_or',
               'bitwise_not', 'any', 'all'}

    def __init__(self):
        super().__init__()
        self.scope = 'trunk'
        self.paused = 0
        self.counts = defaultdict(Counter)
        self.ops = Counter()
        self.unknown = {}
        self.patches = []
        self.hooks = []
        self._scopes = []

    def add(self, **counts):
        self.counts[self.scope].update({k: int(v) for k, v in counts.items()})

    @contextmanager
    def in_scope(self, name):
        previous = self.scope
        self.scope = name
        try:
            yield
        finally:
            self.scope = previous

    @contextmanager
    def suspend(self):
        self.paused += 1
        try:
            yield
        finally:
            self.paused -= 1

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        if self.paused:
            return result
        name = func._schema.name.split('::')[-1]
        key = str(func)
        self.ops[key] += 1
        ins, outs = list(tensors(args)), list(tensors(result))
        inp = ins[0] if ins else None
        out = outs[0] if outs else None
        n = out.numel() if out is not None else 0
        floating = any(t.is_floating_point() for t in outs)
        if name in self.MOVEMENT:
            if name in {'copy_', 'clone', '_to_copy', 'cat', 'index', 'index_select', 'gather'}:
                self.add(copy_output_bytes=sum(t.numel()*t.element_size() for t in outs))
        elif name in self.COMPARE:
            self.add(predicate_outputs=n)
        elif name in {'bitwise_left_shift', 'bitwise_right_shift'}:
            self.add(integer_arithmetic=n)
        elif name in {'mm', 'bmm', 'matmul', 'dot', 'mv'}:
            self.add(flops=matmul_flops(args[0].shape, args[1].shape), matmul_calls=1)
        elif name in {'addmm', 'baddbmm'}:
            beta = kwargs.get('beta', 1)
            alpha = kwargs.get('alpha', 1)
            self.add(flops=matmul_flops(args[1].shape, args[2].shape)
                + (n if beta else 0) + (n if beta not in (0, 1) else 0)
                + (n if alpha != 1 else 0), matmul_calls=1)
        elif name in {'add', 'add_', 'sub', 'sub_', 'mul', 'mul_', 'div', 'div_',
                      'true_divide', 'neg', 'neg_', 'abs', 'abs_', 'reciprocal'}:
            self.add(**{('flops' if floating else 'integer_arithmetic'): n})
        elif name in {'addcmul', 'addcmul_'}:
            self.add(flops=n*(2 if kwargs.get('value', 1) == 1 else 3))
        elif name in {'floor_divide', 'remainder', 'fmod'}:
            self.add(**{('flops' if floating else 'integer_arithmetic'): n})
        elif name in {'sum', 'mean'}:
            amount = inp.numel() - n + (n if name == 'mean' else 0)
            self.add(**{('flops' if floating else 'integer_arithmetic'): amount})
        elif name == 'cumsum':
            self.add(**{('flops' if floating else 'integer_arithmetic'): inp.numel()})
        elif name in {'pow', 'pow_'}:
            if len(args) > 1 and isinstance(args[1], (int, float)) and args[1] == 2:
                self.add(flops=n)
            else:
                self.add(pow=n)
        elif name in {'sqrt', 'rsqrt', 'exp', 'exp2', 'log', 'log2', 'log1p',
                      'erf', 'erfc', 'sin', 'cos', 'tanh', 'floor', 'ceil', 'round', 'trunc'}:
            self.add(**{name: n})
        elif name in {'sigmoid', 'silu'}:
            self.add(flops=(3 if name == 'sigmoid' else 4)*n, exp=n)
        elif name in {'_softmax', 'softmax'}:
            dim = args[1] if len(args) > 1 else kwargs.get('dim', -1)
            self.add(**softmax_work(n, n//inp.shape[dim]))
        elif name == 'linalg_vector_norm':
            order = args[1] if len(args) > 1 else kwargs.get('ord', 2)
            if order == 2:
                self.add(flops=2*inp.numel()-n, sqrt=n)
            else:
                self.unknown[key] = {'ord': order}
        elif name in {'amax', 'amin', 'max', 'min', 'argmax', 'argmin'}:
            self.add(fp_comparisons=max(0, inp.numel()-n))
        elif name in {'clamp', 'clamp_min', 'clamp_max', 'maximum', 'minimum'}:
            self.add(fp_comparisons=n*(2 if name == 'clamp' else 1))
        elif name in {'sort', 'argsort', 'topk'}:
            self.add(sort_input_items=inp.numel(), sort_output_items=n, sort_calls=1)
        elif 'scaled_dot_product' in name and 'backward' not in name:
            self._sdpa(func, args, kwargs)
        else:
            self.unknown[key] = {'inputs': [list(t.shape) for t in ins],
                                 'outputs': [list(t.shape) for t in outs]}
        return result

    def _sdpa(self, func, args, kwargs):
        values = {s.name: v for s, v in zip(func._schema.arguments, args)}
        values.update(kwargs)
        q, k, v = args[:3]
        b, h, nq, d = q.shape
        nk, dv = k.shape[-2], v.shape[-1]
        slots = b*h*nq*nk
        # Nominal dense tensor shapes, plus the exact allowed mask support.
        causal = bool(values.get('is_causal', False))
        mask = values.get('attn_bias', values.get('attn_mask'))
        with self.suspend():
            if mask is not None:
                allow = mask if mask.dtype == torch.bool else torch.isfinite(mask)
                allow = allow.expand(b, h, nq, nk)
                if causal:
                    allow = allow & (torch.arange(nq,device=q.device)[:,None]
                                     >= torch.arange(nk,device=q.device)[None])
                valid = int(allow.sum().item())
            elif causal:
                valid = b*h*sum(min(i+1,nk) for i in range(nq))
            else:
                valid = slots
        self.add(flops=2*slots*(d+dv)+slots, sdpa_shape_pairs=slots,
                 sdpa_valid_pairs=valid, sdpa_qk_av_valid_flops=2*valid*(d+dv))
        self.add(**softmax_work(slots,b*h*nq))

    def _patch(self, obj, name, counter):
        original = getattr(obj, name)

        @wraps(original)
        def call(*args, **kwargs):
            if self.paused:
                return original(*args, **kwargs)
            counter(*args, **kwargs)
            with self.suspend():
                return original(*args, **kwargs)

        self.patches.append((obj,name,original))
        setattr(obj,name,call)

    def install_bounded_kernels(self, *, include_readout=True):
        from src.utils import trunk_triton as tt
        from src.fsattn import cell_select_triton as ct

        def norm(x, weight, eps, dtype, res=None):
            rows=x.numel()//x.shape[-1]
            self.add(flops=4*x.numel()+rows+(x.numel() if res is not None else 0),
                     rsqrt=rows, custom_rms_calls=1)

        def silu(g,u,out=None):
            self.add(flops=5*g.numel(),exp=g.numel(),custom_swiglu_calls=1)

        def read(q,k,v,selected_cells,*,cell_size,query_block,scale,bias=None,q_pos=0):
            # Match the kernel's causal and duplicate predicates. Audit work is
            # outside the ledger and is never used as a timing measurement.
            with self.suspend():
                ids=selected_cells.to(torch.int64)
                pos=torch.arange(q.shape[2],device=q.device)+q_pos
                ok=(ids>=0)&(ids<math.ceil(k.shape[2]/cell_size))
                ok=ok&((ids+1)*cell_size <= (pos//query_block*query_block)[None,None,:,None])
                for j in range(ids.shape[-1]):
                    if j:
                        ok[...,j] &= ~((ids[...,:j] == ids[...,j,None]) & ok[...,:j]).any(-1)
                cells=int(ok.sum().item())
                active=int(ok.any(-1).sum().item())
            pairs=cells*cell_size
            self.add(flops=4*pairs*q.shape[-1]+pairs+(pairs if bias is not None else 0),
                     selected_cells=cells,lr_token_pairs=pairs,custom_lr_calls=1)
            self.add(**softmax_work(pairs,active))

        self._patch(tt,'rms_norm_cast',norm)
        self._patch(tt,'silu_mul',silu)
        if include_readout:
            self._patch(ct,'indexed_cell_attention',read)

    def attach_model(self, model):
        # FSAttn.forward is used by bounded prefill. Graph bodies call explicit
        # attention functions and use in_scope() in the collecting driver.
        modules = [(model.lm_head, 'output_head')]
        for block in model.blocks:
            modules.extend([(block.attn,'attention'),(block.ffn,'mlp')])
            a = block.attn.fsattn
            modules.extend((m,'attention_projection') for m in (a.W_q,a.W_k,a.W_v,a.W_o))
        for module, label in modules:
            def pre(mod, args, name=label):
                self._scopes.append(self.scope)
                self.scope=name
            def post(mod,args,output):
                self.scope=self._scopes.pop()
            self.hooks.append(module.register_forward_pre_hook(pre))
            self.hooks.append(module.register_forward_hook(post, always_call=True))

    def close(self):
        for obj,name,original in reversed(self.patches):
            setattr(obj,name,original)
        for hook in self.hooks:
            hook.remove()
        self.patches.clear()
        self.hooks.clear()

    def report(self):
        all_counts=Counter()
        for counts in self.counts.values():
            all_counts.update(counts)
        return dict(convention='FMA=2; shape arithmetic; transcendentals and comparisons separate',
            total=dict(all_counts),by_scope={k:dict(v) for k,v in self.counts.items()},
            operator_calls=dict(self.ops),unknown_operators=self.unknown,
            complete_operator_coverage=not self.unknown)
