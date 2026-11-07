"""Check closed-form counts against explicit scalar implementations."""
from collections import Counter
import math

from experiments.rebuttal.custom_cost_formulas import (
    cell_statistics, fused_prefill_readout, quantize_cells,
)


class Scalar:
    counts=Counter()

    def __init__(self,value): self.value=float(value)
    @staticmethod
    def value_of(x): return x.value if isinstance(x,Scalar) else x
    def op(self,other,fn):
        self.counts['flops']+=1
        return Scalar(fn(self.value,self.value_of(other)))
    def __add__(self,other):return self.op(other,lambda x,y:x+y)
    def __sub__(self,other):return self.op(other,lambda x,y:x-y)
    def __mul__(self,other):return self.op(other,lambda x,y:x*y)
    def __truediv__(self,other):return self.op(other,lambda x,y:x/y)
    def __abs__(self):
        self.counts['flops']+=1
        return Scalar(abs(self.value))
    def __gt__(self,other):
        self.counts['fp_comparisons']+=1
        return self.value>self.value_of(other)
    def __eq__(self,other):
        self.counts['fp_comparisons']+=1
        return self.value==self.value_of(other)
    def special(self,name,fn):
        self.counts[name]+=1
        return Scalar(fn(self.value))


def reduction(values):
    result=values[0]
    for value in values[1:]:result=result+value
    return result


def matrix_dot(a,b):
    # A matrix product's accumulator convention is FMA=2, including the
    # first multiplication accumulated into zero.
    result=Scalar(0)
    for x,y in zip(a,b):result=result+x*y
    return result


def test_cell_statistics_matches_scalar_mean_and_norms():
    cells=[[[Scalar(1+c+s+d) for d in range(3)] for s in range(4)] for c in range(2)]
    Scalar.counts.clear()
    for cell in cells:
        means=[reduction(list(column))/len(cell) for column in zip(*cell)]
        norms=[reduction([x*x for x in row]).special('sqrt',math.sqrt) for row in cell]
        assert means and max(norms).value>0
    assert dict(Scalar.counts)==cell_statistics(2,4,3)


def test_quantization_matches_scalar_operations():
    cells=[[Scalar(0.25+c+i) for i in range(12)] for c in range(2)]
    Scalar.counts.clear()
    for cell in cells:
        maximum=max(abs(x) for x in cell)
        scale=maximum/127
        if not maximum>0:scale=Scalar(1)
        quantized=[(x/scale).special('round',round) for x in cell]
        assert len(quantized)==12
    assert dict(Scalar.counts)==quantize_cells(2,4,3)


def test_fused_readout_matches_explicit_online_softmax_and_norm():
    # Two query groups, respectively two and one union cells. The second
    # group demonstrates that union work must use its observed population.
    group,size,width=2,2,3
    Scalar.counts.clear()
    for n_union in [2,1]:
        for row in range(group):
            q=[Scalar(0.05*(1+row+d)) for d in range(width)]
            acc=[Scalar(0) for _ in range(width)]
            denominator=Scalar(0)
            maximum=Scalar(-math.inf)
            for cell in range(n_union):
                keys=[[Scalar(0.1*(1+cell+s+d)) for d in range(width)] for s in range(size)]
                values=[[Scalar(0.2*(1+cell+s+d)) for d in range(width)] for s in range(size)]
                scores=[matrix_dot(q,key)*0.5 for key in keys]
                new_max=max(maximum,max(scores))
                used_max=Scalar(0) if new_max==-math.inf else new_max
                alpha=(maximum-used_max).special('exp',math.exp)
                weights=[(score-used_max).special('exp',math.exp) for score in scores]
                denominator=denominator*alpha+reduction(weights)
                acc=[old*alpha+matrix_dot(weights,list(column))
                     for old,column in zip(acc,zip(*values))]
                maximum=new_max
            active=denominator>0
            out=[value/denominator if active else Scalar(0) for value in acc]
            mean_square=reduction([x*x for x in out])*(1/width)
            inv_rms=(mean_square+1e-8).special('rsqrt',lambda x:1/math.sqrt(x))
            normalized=[(x*inv_rms)*0.9 for x in out]
            combined=[Scalar(0.2)+x*0.8 for x in normalized]
            assert all(math.isfinite(x.value) for x in combined)
    assert dict(Scalar.counts)==fused_prefill_readout(3,4,group,size,width)
