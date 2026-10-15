import torch

from experiments.scaling.retention_loss import repeat_sources, retention_margin
from experiments.scaling.train_dist import far_recall_masks


def test_sources_match_existing_labels_and_do_not_cross_documents():
    w = torch.arange(30, 930).unsqueeze(0)
    w[:, 750:770] = w[:, 30:50]
    x, y = w[:, :-1], w[:, 1:]
    src = repeat_sources(x, y)
    far, _ = far_recall_masks(x, y)
    assert torch.equal(src >= 0, far)
    assert (src >= 0).any()
    assert torch.equal(w.gather(1, src.clamp_min(0))[src >= 0], y[src >= 0])
    w[:, 400] = 2
    assert not (repeat_sources(w[:, :-1], w[:, 1:]) >= 0).any()


def test_margin_pushes_source_only_and_ignores_future_cells():
    keys = torch.tensor([[[1.], [1.], [3.], [3.], [4.], [4.], [100.], [100.]]], requires_grad=True)
    sources = torch.full((1,8), -1)
    sources[0,6] = 0
    loss, count = retention_margin(keys,sources,heads=1,stride=2,block=2,rank=2,margin=.1)
    assert count == 1
    torch.testing.assert_close(loss, torch.tensor(1.1 - 1/3))
    loss.backward()
    assert keys.grad[0,:2].sum() < 0
    assert keys.grad[0,2:].count_nonzero() == 0


def test_no_targets_and_early_sources_are_differentiable_zero():
    keys = torch.randn(1,8,2,requires_grad=True)
    for early in (False, True):
        sources = torch.full((1,8),-1)
        if early:
            sources[0,2] = 0
        loss,count = retention_margin(keys,sources,heads=1,stride=2,block=2,rank=2,margin=.02)
        assert loss == 0 and count == 0
        loss.backward()
    assert keys.grad.count_nonzero() == 0


def test_well_retained_source_has_no_penalty_and_duplicates_do_not_reweight():
    keys = torch.tensor([[[4.],[4.],[1.],[1.],[2.],[2.],[1.],[1.]]])
    sources = torch.full((1,8),-1)
    sources[0,6:] = 0
    loss,count = retention_margin(keys,sources,heads=1,stride=2,block=2,rank=2,margin=.02)
    assert loss == 0 and count == 1
