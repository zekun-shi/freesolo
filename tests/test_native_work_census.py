"""Check readout accounting against explicit causal key/query membership."""
import torch

from experiments.rebuttal.native_work_census import readout_work


def test_readout_union_counts_include_unselected_tile_rows():
    # At query positions 8..11, three distinct cells serve four rows. The
    # fused kernel computes all 4x3 pairs before masking each row's membership.
    cells=torch.tensor([[[[-1,-1]]*8+[[0,0],[1,-1],[2,0],[2,1]]+[[0,3]]*4]])
    got=readout_work(cells,16,16,2,8,4)
    logical=0;unions=0
    for start in range(0,16,4):
        union=set()
        for position in range(start,start+4):
            allowed={int(c) for c in cells[0,0,position]
                if 0<=int(c)<8 and (int(c)+1)*2<=position//8*8}
            logical+=len(allowed)*2
            union.update(allowed)
        unions+=len(union)
    assert got['readout_logical_token_pairs']==logical==28
    assert got['readout_tile_token_pairs']==unions*4*2==40
    assert got['readout_group_union_cells']==5
    assert got['readout_sort_items']==32


def test_readout_rejects_future_and_out_of_range_cells():
    cells=torch.tensor([[[[0,1],[0,-1],[7,2],[1,1],[0,1],[2,7],[0,0],[1,-1]]]])
    got=readout_work(cells,8,8,2,4,2)
    assert got['readout_selected_cells']==4
    assert got['readout_group_union_cells']==4
    assert got['readout_logical_token_pairs']==8
    assert got['readout_tile_token_pairs']==16
