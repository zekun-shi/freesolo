"""Check counted chunk GSA against its recurrent mathematical definition."""
from collections import Counter
import math

from experiments.rebuttal.baseline_cost import (
    gsa_chunk_components,gsa_recurrent,dense_attention,model_cost,raven_outer,
)
from test_custom_cost_formulas import Scalar,reduction,matrix_dot


def zeros(rows,cols):return [[Scalar(0) for _ in range(cols)] for _ in range(rows)]
def transpose(x):return list(map(list,zip(*x)))
def dot(a,b,initial=None):
    out=Scalar(0) if initial is None else initial
    for x,y in zip(a,b):out=out+x*y
    return out
def mm(a,b,initial=None):
    return [[dot(row,col,None if initial is None else initial[i][j])
             for j,col in enumerate(transpose(b))] for i,row in enumerate(a)]
def exp2(x):return x.special('exp2',lambda z:2**z)
def softmax(row):
    maximum=max(row)
    weights=[(x-maximum).special('exp',math.exp) for x in row]
    total=reduction(weights)
    return [x/total for x in weights]


def chunk_reference(q,k,v,s,g,b,c):
    length,key,value,slots=len(q),len(q[0]),len(v[0]),len(s[0])
    hk=zeros(key,slots);hv=zeros(slots,value);outputs=[];components={}
    def capture(name,before):
        counts=Counter({a:n-before[a] for a,n in Scalar.counts.items() if n-before[a]})
        components.setdefault(name,Counter()).update(counts)
    for start in range(0,length,b):
        qq,kk,vv,ss,gg=[a[start:start+b] for a in [q,k,v,s,g]]
        before=Scalar.counts.copy();gc=zeros(b,slots)
        for m in range(slots):
            acc=None
            for t in range(b):
                acc=gg[t][m] if acc is None else acc+gg[t][m]
                gc[t][m]=acc*(1/math.log(2))
        capture('local_log_gate_prefix',before)

        before=Scalar.counts.copy()
        decay=[exp2(x) for x in gc[-1]]
        weighted=[[ss[t][m]*exp2(gc[-1][m]-gc[t][m]) for m in range(slots)] for t in range(b)]
        hknew=mm(transpose(kk),weighted,[[hk[d][m]*decay[m] for m in range(slots)] for d in range(key)])
        capture('key_state',before)

        before=Scalar.counts.copy()
        qs=[[x*1.0 for x in row] for row in qq]
        inter=mm(qs,hk);a=mm(qs,transpose(kk))
        inter=[[inter[t][m]*exp2(gc[t][m]) for m in range(slots)] for t in range(b)]
        for t in range(b):
            for u in range(t+1,b):a[t][u]=Scalar(0)
        capture('key_inter',before)

        before=Scalar.counts.copy();ok=zeros(b,slots)
        for i in range(b//c):
            off=i*c;acc=zeros(c,slots);anchor=gc[off]
            for j in range(i):
                joff=j*c
                gated=[[ss[joff+t][m]*exp2(anchor[m]-gc[joff+t][m]) for m in range(slots)] for t in range(c)]
                acc=mm([r[joff:joff+c] for r in a[off:off+c]],gated,acc)
            acc=[[acc[t][m]*exp2(gc[off+t][m]-anchor[m]) for m in range(slots)] for t in range(c)]
            for j in range(c):
                for t in range(c):
                    for m in range(slots):
                        gated=ss[off+j][m]*exp2(gc[off+t][m]-gc[off+j][m])
                        contribution=a[off+t][off+j]*gated
                        acc[t][m]=acc[t][m]+(contribution if t>=j else Scalar(0))
            for t in range(c):
                ok[off+t]=[acc[t][m]+inter[off+t][m] for m in range(slots)]
        capture('key_intra',before)

        before=Scalar.counts.copy();qv=[softmax(row) for row in ok]
        capture('slot_softmax',before)

        before=Scalar.counts.copy()
        decay=[exp2(x) for x in gc[-1]]
        weighted=[[ss[t][m]*exp2(gc[-1][m]-gc[t][m]) for m in range(slots)] for t in range(b)]
        hvnew=mm(transpose(weighted),vv,[[hv[m][d]*decay[m] for d in range(value)] for m in range(slots)])
        capture('value_state',before)

        before=Scalar.counts.copy();av=zeros(b,b)
        for i in range(b//c):
            off=i*c;anchor=gc[off]
            for j in range(i):
                joff=j*c
                qg=[[qv[off+t][m]*exp2(gc[off+t][m]-anchor[m])*1.0 for m in range(slots)] for t in range(c)]
                kg=[[ss[joff+t][m]*exp2(anchor[m]-gc[joff+t][m]) for m in range(slots)] for t in range(c)]
                tile=mm(qg,transpose(kg))
                for t in range(c):av[off+t][joff:joff+c]=tile[t]
            for j in range(c):
                for t in range(c):
                    terms=[qv[off+t][m]*ss[off+j][m]*exp2(gc[off+t][m]-gc[off+j][m]) for m in range(slots)]
                    value_at=reduction(terms)*1.0
                    av[off+t][off+j]=value_at if t>=j else Scalar(0)
        capture('value_intra',before)

        before=Scalar.counts.copy()
        qg=[[qv[t][m]*exp2(gc[t][m]) for m in range(slots)] for t in range(b)]
        out=mm(qg,hv)
        out=[[x*1.0 for x in row] for row in out]
        out=mm(av,vv,out)
        capture('value_output',before)
        outputs.extend(out);hk,hv=hknew,hvnew
    return outputs,hk,hv,{k:dict(v) for k,v in components.items()}


def recurrence_float(q,k,v,s,g):
    key,value,slots=len(q[0]),len(v[0]),len(s[0])
    hk=[[0.0]*slots for _ in range(key)];hv=[[0.0]*value for _ in range(slots)];out=[]
    for qq,kk,vv,ss,gg in zip(q,k,v,s,g):
        decay=[math.exp(x) for x in gg]
        hk=[[hk[d][m]*decay[m]+kk[d]*ss[m] for m in range(slots)] for d in range(key)]
        ok=[sum(qq[d]*hk[d][m] for d in range(key)) for m in range(slots)]
        mx=max(ok);p=[math.exp(x-mx) for x in ok];z=sum(p);p=[x/z for x in p]
        hv=[[hv[m][d]*decay[m]+ss[m]*vv[d] for d in range(value)] for m in range(slots)]
        out.append([sum(p[m]*hv[m][d] for m in range(slots)) for d in range(value)])
    return out,hk,hv


def test_chunk_cost_and_numerical_equivalence_to_recurrence():
    length,key,value,slots,b,c=8,2,3,3,4,2
    arrays=[[[0.02*(1+t+d) for d in range(w)] for t in range(length)] for w in [key,key,value,slots,slots]]
    arrays[-1]=[[-x for x in row] for row in arrays[-1]]
    wrapped=[[[Scalar(x) for x in row] for row in a] for a in arrays]
    Scalar.counts.clear()
    out,hk,hv,components=chunk_reference(*wrapped,b,c)
    assert components==gsa_chunk_components(length,1,key,value,slots,b,c)
    reference=recurrence_float(*arrays)
    for actual,expected in zip([out,hk,hv],reference):
        for ar,er in zip(actual,expected):
            assert all(math.isclose(a.value,e,rel_tol=1e-12,abs_tol=1e-12) for a,e in zip(ar,er))


def test_recurrent_cost_counts_both_states_and_one_slot_softmax():
    rows,key,value,slots=2,3,2,4
    Scalar.counts.clear()
    for _ in range(rows):
        q=[Scalar(0.1)*1 for _ in range(key)]
        decay=[Scalar(-0.2).special('exp',math.exp) for _ in range(slots)]
        hk=[[Scalar(0.3)*decay[m]+Scalar(0.2)*Scalar(0.4) for m in range(slots)] for _ in range(key)]
        scores=[reduction([q[d]*hk[d][m] for d in range(key)]) for m in range(slots)]
        p=softmax(scores)
        hv=[[Scalar(0.3)*decay[m]+Scalar(0.4)*Scalar(0.2) for _ in range(value)] for m in range(slots)]
        out=[reduction([p[m]*hv[m][d] for m in range(slots)]) for d in range(value)]
        assert out
    assert dict(Scalar.counts)==gsa_recurrent(rows,key,value,slots)


def test_dense_attention_includes_causal_prefix_and_all_decode_positions():
    assert dense_attention(2,4,8)['causal_token_pairs']==2*sum(range(1,9))
    assert dense_attention(2,4,8,steps=3)['causal_token_pairs']==2*sum(range(9,12))


def test_output_head_standardization_and_raven_prefill_mode():
    for arm in ['tfpp','ravenfull']:
        last=model_cost(arm,4096)
        all_positions=model_cost(arm,4096,head_positions=4096)
        assert all_positions['total']['flops']-last['total']['flops']==4095*2*1024*32000
        assert all_positions['attention']==last['attention']
    pre=model_cost('ravenfull',4096)
    dec=model_cost('ravenfull',4096,steps=257)
    assert 'raven_key_state' in pre['components'] and 'raven_recurrence' not in pre['components']
    assert 'raven_recurrence' in dec['components'] and 'raven_key_state' not in dec['components']
    assert raven_outer(3,16,2,4,2,parameter_evaluations=3)['exp']-raven_outer(3,16,2,4,2)['exp']==4
