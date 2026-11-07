"""Check closed-form counts against explicit scalar implementations."""
from collections import Counter
import math

from experiments.rebuttal.custom_cost_formulas import (
    cell_statistics, fused_prefill_readout, quantize_cells, short_decode_int8,
    streaming_decode_readout_int8, flat_decode_readout_int8,
    forest_node, spine_merge,
    pooled_decode_selection, beam_node_scores, candidate_lse_scores,
    dense_prefill_selection, pooled_prefill_selection,
    dense_decode_selection,
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


def norm(values):
    mean_square=reduction([x*x for x in values])*(1/len(values))
    inv_rms=(mean_square+1e-8).special('rsqrt',lambda x:1/math.sqrt(x))
    return [(x*inv_rms)*0.9 for x in values]


def rotate(values):
    half=len(values)//2
    left=[values[i]*0.9-values[i+half]*0.1 for i in range(half)]
    right=[values[i+half]*0.9+values[i]*0.1 for i in range(half)]
    return left+right


def test_local_decode_matches_dequantize_rotate_gemm_and_softmax():
    rows,window,width=2,3,4
    Scalar.counts.clear()
    for _ in range(rows):
        q=rotate([Scalar(0.1+d) for d in range(width)])
        keys=[rotate([Scalar(1+s+d)*0.1 for d in range(width)]) for s in range(window)]
        values=[[Scalar(2+s+d)*0.1 for d in range(width)] for s in range(window)]
        scores=[matrix_dot(q,k) for k in keys]
        scores=[score*0.5 for score in scores]
        for s in range(window):
            assert (abs(Scalar(window-1)-Scalar(s))>window) is False
        maximum=max(scores)
        weights=[(score-maximum).special('exp',math.exp) for score in scores]
        denominator=reduction(weights)
        weights=[x/denominator for x in weights]
        out=[matrix_dot(weights,list(column)) for column in zip(*values)]
        assert all(math.isfinite(x.value) for x in out)
    assert dict(Scalar.counts)==short_decode_int8(rows,window,width)


def test_streaming_decode_counts_explicit_reductions_and_online_rescaling():
    rows,cells,size,width=2,2,3,4
    Scalar.counts.clear()
    for row in range(rows):
        q=[Scalar(0.05*(1+row+d)) for d in range(width)]
        acc=[Scalar(0) for _ in range(width)]
        denominator=Scalar(0)
        maximum=Scalar(-math.inf)
        for cell in range(cells):
            keys=[[Scalar(1+cell+s+d)*0.1 for d in range(width)] for s in range(size)]
            values=[[Scalar(2+cell+s+d)*0.1 for d in range(width)] for s in range(size)]
            scores=[reduction([a*b for a,b in zip(q,k)])*0.5 for k in keys]
            new_max=max(maximum,max(scores))
            alpha=(maximum-new_max).special('exp',math.exp)
            weights=[(score-new_max).special('exp',math.exp) for score in scores]
            denominator=denominator*alpha+reduction(weights)
            acc=[old*alpha+reduction([a*b for a,b in zip(weights,column)])
                 for old,column in zip(acc,zip(*values))]
            maximum=new_max
        assert denominator>0
        out=[x*0.8 for x in norm([x/denominator for x in acc])]
        assert all(math.isfinite(x.value) for x in out)
    assert dict(Scalar.counts)==streaming_decode_readout_int8(rows,cells,size,width)


def test_flat_decode_counts_one_softmax_over_all_selected_cells():
    rows,cells,size,width=2,2,3,4
    Scalar.counts.clear()
    for row in range(rows):
        q=[Scalar(0.05*(1+row+d)) for d in range(width)]
        keys=[[Scalar(1+s+d)*0.1 for d in range(width)] for s in range(cells*size)]
        values=[[Scalar(2+s+d)*0.1 for d in range(width)] for s in range(cells*size)]
        scores=[reduction([a*b for a,b in zip(q,k)])*0.5 for k in keys]
        maximum=max(scores)
        weights=[(score-maximum).special('exp',math.exp) for score in scores]
        denominator=reduction(weights)
        weights=[x/denominator for x in weights]
        out=norm([reduction([a*b for a,b in zip(weights,column)]) for column in zip(*values)])
        assert all(math.isfinite(x.value) for x in out)
    assert dict(Scalar.counts)==flat_decode_readout_int8(rows,cells,size,width)


def unit(values):
    denominator=max(reduction([x*x for x in values]).special('sqrt',math.sqrt),Scalar(1e-6))
    return [x/denominator for x in values]


def test_forest_node_counts_complete_split_algebra():
    for rows in [1,2,4,8]:
        width=3
        points=[[Scalar(1+r*r+d*0.2*r) for d in range(width)] for r in range(rows)]
        ceilings=[Scalar(2+r) for r in range(rows)]
        Scalar.counts.clear()
        means=[reduction(list(column))/rows for column in zip(*points)]
        # Quantization itself is a cast, not a floating arithmetic operation.
        rounded=[Scalar(round(x.value,2)) for x in means]
        rounding_error=[a-b for a,b in zip(means,rounded)]
        slack=reduction([x*x for x in rounding_error]).special('sqrt',math.sqrt)
        if rows==1:
            radius=Scalar(0)+slack
            ceiling=max(ceilings[0],Scalar(0))
        else:
            centered=[[x-mu for x,mu in zip(point,means)] for point in points]
            norms=[reduction([x*x for x in point]) for point in centered]
            radius=max(norms).special('sqrt',math.sqrt)+slack
            ceiling=max(max(ceilings),Scalar(0))
            far=max(range(rows),key=lambda i:norms[i])
            denominator=max(norms[far].special('sqrt',math.sqrt),Scalar(1e-6))
            direction=[x/denominator for x in centered[far]]
            for _ in range(2):
                projection=[reduction([x*v for x,v in zip(point,direction)]) for point in centered]
                direction=unit([reduction([x*p for x,p in zip(column,projection)])
                                for column in zip(*centered)])
            projection=[reduction([x*v for x,v in zip(point,direction)]) for point in centered]
            Scalar.counts['sort_input_items']+=len(projection)
        actual=dict(Scalar.counts)
        actual.setdefault('sort_input_items',0)
        assert math.isfinite(radius.value+ceiling.value)
        assert actual==forest_node(rows,width)


def test_spine_merge_counts_weighted_mean_and_enclosing_radius():
    width=4
    Scalar.counts.clear()
    na,nb=Scalar(3),Scalar(5)
    a=[Scalar(i) for i in range(width)]
    b=[Scalar(2+i) for i in range(width)]
    total=na+nb
    mean=[(x*na+y*nb)/total for x,y in zip(a,b)]
    distances=[]
    for center in [a,b]:
        delta=[x-m for x,m in zip(center,mean)]
        distances.append(reduction([x*x for x in delta]).special('sqrt',math.sqrt))
    radius=max(Scalar(0.7)+distances[0],Scalar(0.8)+distances[1])
    ceiling=max(Scalar(3),Scalar(4))
    assert math.isfinite(radius.value+ceiling.value)
    assert dict(Scalar.counts)==spine_merge(width)


def test_pooled_selection_counts_masked_candidate_vector_and_shared_denominator():
    rows,candidates,width,topk=2,3,64,2
    Scalar.counts.clear()
    for _ in range(rows):
        denominator=Scalar(2)*Scalar(4)
        for c in range(4):
            dot=Scalar(0)
            for t in range(16):
                terms=[Scalar(0.01*(t+i+1))*Scalar(0.02*(c+i+1)) for i in range(4)]
                dot=dot+reduction(terms)
            content=dot*0.125
            distance=abs(Scalar(100)-Scalar(16+32*c))
            score=content+(Scalar(0.7)*Scalar(0.9))*(distance/denominator).special('erf',math.erf)
            assert math.isfinite(score.value)
    actual=dict(Scalar.counts,topk_input_items=8,topk_passes=rows*topk)
    assert actual==pooled_decode_selection(rows,candidates,width,topk)


def test_beam_score_matches_explicit_bound_expression():
    visits,queries,width=3,2,4
    Scalar.counts.clear()
    for query,nodes in enumerate([1,2]):
        q=[Scalar(0.1*(query+d+1)) for d in range(width)]
        qnorm=reduction([x*x for x in q]).special('sqrt',math.sqrt)*8
        denominator=Scalar(2)*Scalar(4)
        for node in range(nodes):
            mu=[Scalar(0.2*(node+d+1)) for d in range(width)]
            content=reduction([x*y for x,y in zip(q,mu)])*0.125
            content=content+(qnorm*Scalar(0.3))*0.125
            center=(Scalar(node)+0.5)*32
            position=(abs(Scalar(128)-center)/denominator).special('erf',math.erf)
            score=content+(Scalar(0.7)*Scalar(0.9))*position
            assert math.isfinite(score.value)
    assert dict(Scalar.counts,beam_scored_nodes=visits)==beam_node_scores(visits,queries,width)


def test_candidate_lse_covers_dequantization_and_duplicate_scoring():
    pairs,size,width=3,2,4
    for int8 in [False,True]:
        Scalar.counts.clear()
        for pair in range(pairs):
            q=[Scalar(0.1*(d+1)) for d in range(width)]
            keys=[[Scalar(0.2*(s+d+1)) for d in range(width)] for s in range(size)]
            if int8:keys=[[x*0.3 for x in key] for key in keys]
            if int8:scores=[reduction([x*y for x,y in zip(q,key)])*0.125 for key in keys]
            else:scores=[matrix_dot(q,key)*0.125 for key in keys]
            maximum=max(scores)
            weights=[(x-maximum).special('exp',math.exp) for x in scores]
            lse=maximum+reduction(weights).special('log',math.log)
            center=(Scalar(pair%2)+0.5)*32
            denominator=Scalar(2)*Scalar(4)
            positional=(abs(Scalar(128)-center)/denominator).special('erf',math.erf)
            score=lse+(Scalar(0.7)*Scalar(0.9))*positional
            assert math.isfinite(score.value)
        assert dict(Scalar.counts)==candidate_lse_scores(pairs,size,width,int8=int8)


def test_dense_selection_counts_partial_tiles_and_far_shortcut():
    heads,blocks,bq,size,width,tile=2,7,2,2,3,4
    sigma=0.2
    threshold=3.9194984436035156
    Scalar.counts.clear()
    erfs=items=pairs=0
    for _ in range(heads):
        for block in range(1,blocks):
            visible=block*bq//size
            start=Scalar(block)*bq
            denominator=Scalar(sigma)+Scalar(sigma)
            margin=denominator*threshold
            positions=[Scalar(i)+start for i in range(bq)]
            for c0 in range(0,visible,tile):
                nc=min(tile,visible-c0)
                far=False
                if nc==tile:
                    distance=start-(Scalar(c0+tile-1)+0.5)*size
                    far=distance.value>=margin.value
                bonus=[];centers=[];near=[]
                for c in range(nc):
                    if not far:
                        center=(Scalar(c0+c)+0.5)*size
                        distance=start-center
                        centers.append(center)
                        near.append(distance.value<margin.value)
                    bonus.append(Scalar(0.7)*Scalar(0.9))
                for pos in positions:
                    for c in range(tile):
                        score=matrix_dot([Scalar(0.1)]*width,[Scalar(0.2)]*width)*0.5
                        score=score+(bonus[c] if c<nc else Scalar(0))
                        pairs+=1;items+=1
                    if not far:
                        for c in range(nc):
                            if near[c]:
                                term=(abs(pos-centers[c])/denominator).special('erf',math.erf)
                                score=Scalar(0.1)+bonus[c]*term
                                erfs+=1;items+=1
    actual=dict(Scalar.counts,selection_score_items=items,selection_gemm_pairs=pairs)
    assert actual==dense_prefill_selection(heads,blocks,bq,size,width,sigma=sigma,tile=tile)


def test_pooled_prefill_counts_cold_block_and_deduplicated_near_cell():
    heads,blocks,bq,size,width,pool,recent=1,3,2,2,3,33,2
    sigma=0.2;candidates=pool+recent
    Scalar.counts.clear();items=pairs=0
    for block in range(blocks):
        visible=block*bq//size
        start=Scalar(block)*bq
        denominator=Scalar(sigma)+Scalar(sigma)
        for c in range(candidates):
            cand=c if c<pool else visible-1-(c-pool)
            cc=max(cand,0)
            if cand>=0:visibility=(Scalar(cc)+1)*size
            center=(Scalar(cc)+0.5)*size
            distance=start-center
            margin=denominator*3.9194984436035156
            bonus=Scalar(0.7)*Scalar(0.9)
        for query in range(bq):
            pos=Scalar(query)+start
            for c in range(candidates):
                score=matrix_dot([Scalar(0.1)]*width,[Scalar(0.2)]*width)*0.5
                score=score+Scalar(0.2)
                pairs+=1;items+=1
            # The last visible cell survives once after top-P/recent dedup.
            if visible:
                term=(abs(pos-Scalar((visible-0.5)*size))/denominator).special('erf',math.erf)
                score=Scalar(0.1)+Scalar(0.2)*term
                items+=1
    actual=dict(Scalar.counts,selection_score_items=items,selection_gemm_pairs=pairs)
    assert actual==pooled_prefill_selection(heads,blocks,bq,size,width,pool,recent,sigma=sigma)


def test_dense_decode_tracks_block_boundary_visibility_and_erf_shortcut():
    heads,length,steps,bq,size,width,splits=2,8,5,4,2,4,2
    sigma=0.2
    Scalar.counts.clear();cells=0
    for _ in range(heads):
        for position in range(length,length+steps):
            for _split in range(splits):
                denominator=Scalar(sigma)+Scalar(sigma)
                margin=denominator*3.9194984436035156
            for c in range((position//bq)*bq//size):
                dot=Scalar(0)
                for t in range(width//4):
                    terms=[Scalar(0.1)*Scalar(0.2) for _ in range(4)]
                    dot=dot+reduction(terms)
                content=dot*0.5
                center=(Scalar(c)+0.5)*size
                bonus=Scalar(0.7)*Scalar(0.9)
                distance=abs(Scalar(position)-center)
                erf=(distance/denominator).special('erf',math.erf) if distance.value<margin.value else Scalar(1)
                score=content+bonus*erf
                cells+=1
    actual=dict(Scalar.counts,selection_score_items=cells)
    assert actual==dense_decode_selection(heads,length,steps,bq,size,width,splits,sigma=sigma)
