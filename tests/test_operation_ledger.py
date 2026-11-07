"""Check counts against explicit scalar loops and actual dispatched operations."""
import pytest
import torch

from experiments.rebuttal.operation_ledger import OperationLedger,matmul_flops


@pytest.mark.parametrize('a,b', [((3,5),(5,7)),((2,3,5),(2,5,7)),((5,),(5,)),
                              ((4,1,3,5),(1,2,5,7))])
def test_matmul_matches_explicit_multiply_accumulate_count(a,b):
    x,y=torch.randn(a),torch.randn(b)
    z=x@y
    manual=sum(2 for _ in range(z.numel()) for _ in range(a[-1]))
    assert matmul_flops(a,b)==manual
    with OperationLedger() as ledger:
        got=x@y
    assert torch.equal(got,z)
    assert ledger.report()['total']['flops']==manual
    assert not ledger.unknown


def test_softmax_and_norm_include_reductions_and_special_functions():
    x=torch.randn(3,7)
    with OperationLedger() as ledger:
        y=x.softmax(-1)
        n=x.norm(dim=-1)
    assert torch.equal(y,x.softmax(-1)) and torch.equal(n,x.norm(dim=-1))
    counts=ledger.report()['total']
    assert counts['flops']==(3*21-3)+(2*21-3)
    assert counts['exp']==21 and counts['sqrt']==3
    assert not ledger.unknown


def test_unknown_arithmetic_prevents_complete_coverage_claim():
    with OperationLedger() as ledger:
        torch.fft.fft(torch.arange(8,dtype=torch.float32))
    assert ledger.unknown
    assert not ledger.report()['complete_operator_coverage']


def test_dtype_promotion_has_no_tensor_work():
    with OperationLedger() as ledger:
        dtype = torch.promote_types(torch.bfloat16, torch.float32)
    assert dtype == torch.float32
    assert ledger.ops['aten.promote_types.default'] == 1
    assert ledger.report()['total'] == {}
    assert not ledger.unknown


def test_rounding_and_index_shifts_are_separate_from_flops():
    x = torch.tensor([-1.5, 0.25, 3.75])
    with OperationLedger() as ledger:
        rounded = x.floor()
        shifted = rounded.to(torch.int64) << 2
        shifted_tensor = torch.bitwise_left_shift(rounded.to(torch.int64), torch.tensor([2,2,2]))
    assert shifted.tolist() == [-8, 0, 12]
    assert torch.equal(shifted, shifted_tensor)
    counts = ledger.report()['total']
    assert counts.get('flops', 0) == 0
    assert counts['floor'] == 3 and counts['integer_arithmetic'] == 6
    assert not ledger.unknown


def test_reverse_subtraction_distinguishes_floating_and_index_arithmetic():
    x = torch.tensor([0.25, 0.5, 0.75])
    indices = torch.tensor([0, 3, 7, 15])
    with OperationLedger() as ledger:
        weights = 1 - x
        reversed_indices = 15 - indices
    assert weights.tolist() == [0.75, 0.5, 0.25]
    assert reversed_indices.tolist() == [15, 12, 8, 0]
    counts = ledger.report()['total']
    assert counts['flops'] == 3 and counts['integer_arithmetic'] == 4
    assert ledger.ops['aten.rsub.Scalar'] == 2
    assert not ledger.unknown


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA indexed attention validation')
def test_custom_read_counts_valid_unique_causal_cells_and_preserves_output():
    from src.fsattn.cell_select_triton import indexed_cell_attention
    import src.fsattn.cell_select_triton as ct
    q=torch.randn(1,2,3,16,device='cuda')
    k=torch.randn(1,2,32,16,device='cuda')
    v=torch.randn_like(k)
    ids=torch.tensor([0,0,1,-1],device='cuda',dtype=torch.int32).expand(1,2,3,4).contiguous()
    kwargs=dict(cell_size=8,query_block=16,scale=.25,q_pos=16)
    reference=indexed_cell_attention(q,k,v,ids,**kwargs)
    ledger=OperationLedger();ledger.install_bounded_kernels()
    try:
        with ledger:
            got=ct.indexed_cell_attention(q,k,v,ids,**kwargs)
    finally:ledger.close()
    assert torch.equal(got,reference)
    counts=ledger.report()['total']
    assert counts['selected_cells']==12
    assert counts['lr_token_pairs']==96
    assert counts['flops']==4*96*16+96+(3*96-6)
    assert not ledger.unknown
