"""Algorithmic costs for the retained TF++ and Raven-full benchmark geometry.

FMA=2, explicit reductions=n-1, scalar transcendental calls separate. Raven
prefill follows chunk GSA (BT=64, BC=16); it is not priced as recurrence.
Shared algebra is counted once, excluding autotuner-dependent duplicate
computations across tiles. The archived timing harness emits logits for all
prefill positions; callers explicitly choose the output-head convention.
"""
from collections import Counter


def gsa_chunk_components(length, heads, key, value, slots, chunk=64, subchunk=16):
    assert length>0 and length%chunk==0 and chunk%subchunk==0
    assert min(heads,key,value,slots)>0 and slots<=256
    b,c,k,v,m=chunk,subchunk,key,value,slots
    groups=b//c;off=groups*(groups-1)//2
    components={
        'local_log_gate_prefix':dict(flops=(2*b-1)*m),
        'key_state':dict(flops=2*b*k*m+2*b*m+k*m,exp2=(b+1)*m),
        'key_inter':dict(flops=b*k+2*b*k*m+2*b*b*k+b*m,exp2=b*m),
        'key_intra':dict(flops=off*(2*c*m+2*c*c*m)+groups*(3*c*m+4*c*c*m),
                         exp2=off*c*m+b*m+b*c*m),
        'slot_softmax':dict(flops=b*(3*m-1),exp=b*m,fp_comparisons=b*(m-1)),
        'value_state':dict(flops=2*b*m*v+2*b*m+m*v,exp2=(b+1)*m),
        'value_intra':dict(flops=off*(5*c*m+2*c*c*m)+4*b*c*m,
                           exp2=2*off*c*m+b*c*m),
        'value_output':dict(flops=b*m+2*b*m*v+b*v+2*b*b*v,exp2=b*m),
    }
    copies=heads*(length//chunk)
    return {name:{unit:count*copies for unit,count in counts.items()} for name,counts in components.items()}


def gsa_recurrent(rows, key, value, slots):
    """The benchmark's pure-Torch two-state recurrence and slot softmax."""
    k,v,m=key,value,slots
    assert min(k,v,m)>0 and rows>=0
    return dict(flops=rows*(5*m*(k+v)+k+2*m-v-1),
                exp=rows*2*m,fp_comparisons=rows*(m-1))


def dense_attention(heads, width, length, steps=None):
    if steps is None:
        rows=heads*length
        pairs=heads*length*(length+1)//2
    else:
        rows=heads*steps
        pairs=heads*(steps*length+steps*(steps+1)//2)
    return dict(flops=pairs*(4*width+4)-rows,exp=pairs,
                fp_comparisons=pairs-rows,causal_token_pairs=pairs)


def raven_outer(tokens, hidden, heads, slots, topk, *, parameter_evaluations=1):
    """Projections, swish/QK norms, router, Mamba2 gate and output swish/norm.

    The two recurrent state updates are priced separately. Bias-free QKVO
    and linear routing match the recovered Raven-full configuration.
    Softplus is retained as its own scalar call, like exp/rsqrt.
    """
    d,h,m=hidden,heads,slots
    assert d%h==0 and topk<=m
    k=d//h
    return dict(
        flops=tokens*(8*d*d+2*d*h*m+2*d*h+16*d+2*h*(4*k+1)+(4*d+1)
                      +3*h*m+2*h*topk+2*h+2*h*m)+h*parameter_evaluations,
        exp=tokens*(4*d+2*h*m)+h*parameter_evaluations,rsqrt=tokens*(2*h+1),softplus=tokens*h,
        topk_input_items=tokens*h*m,topk_output_items=tokens*h*topk)


def model_cost(arm, length, *, steps=None, head_positions=1):
    """Standardized model arithmetic; prefill head_positions is explicit."""
    assert arm in ['tfpp','ravenfull'] and length>0
    d,ff,vocab=1024,2816,32000
    layers=24 if arm=='tfpp' else 21
    tokens=length if steps is None else steps
    output_rows=head_positions if steps is None else steps
    assert 0<output_rows<=tokens
    pieces={
        'mlp':dict(flops=layers*tokens*(6*d*ff+5*ff),exp=layers*tokens*ff),
        'block_norms_residuals':dict(flops=layers*tokens*(10*d+2),rsqrt=2*layers*tokens),
        'final_norm':dict(flops=tokens*(4*d+1),rsqrt=tokens),
        'output_head':dict(flops=output_rows*2*d*vocab),
    }
    if arm=='tfpp':
        pieces['attention_projection_rope']=dict(flops=layers*tokens*(8*d*d+6*d))
        pieces['attention_core']={k:v*layers for k,v in dense_attention(16,64,length,steps).items()}
    else:
        pieces['global_attention_projection']=dict(flops=2*tokens*8*d*d)
        pieces['global_attention_core']={k:v*2 for k,v in dense_attention(16,64,length,steps).items()}
        pieces['raven_outer']={k:v*19 for k,v in raven_outer(tokens,d,4,256,32,
            parameter_evaluations=1 if steps is None else steps).items()}
        if steps is None:
            pieces.update({'raven_'+name:{k:v*19 for k,v in count.items()}
                for name,count in gsa_chunk_components(length,4,256,256,256).items()})
        else:
            pieces['raven_recurrence']={k:v*19 for k,v in gsa_recurrent(steps*4,256,256,256).items()}
    total=Counter();attention=Counter()
    for name,count in pieces.items():
        total.update(count)
        if name not in ['mlp','block_norms_residuals','final_norm','output_head']:attention.update(count)
    return dict(arm=arm,length=length,steps=steps,head_positions=output_rows,
        total=dict(total),attention=dict(attention),components=pieces,
        geometry=dict(hidden=d,ffn=ff,vocabulary=vocab,layers=layers),
        scope='Algorithmic arithmetic, causal dense pairs, logical shared chunk algebra; not hardware instructions')
