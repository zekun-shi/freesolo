"""Natural-repeat supervision for the norm-ranked FS-L retention pool."""

import torch
import torch.nn.functional as F


@torch.no_grad()
def repeat_sources(x, y, *, dmin=512, eos=2):
    from experiments.scaling.train_dist import _gram_codes, _prev_unique
    w = torch.cat((x, y[:, -1:]), 1)
    batch, length = w.shape
    n = length - 1
    t = torch.arange(n, device=w.device)
    e = (w == eos).long()
    doc = e.cumsum(1) - e
    p = _prev_unique(_gram_codes(w, 4))[:, (t - 3).clamp_min(0)]
    pc = p.clamp_min(0)
    src = pc + 4
    cont = ((t >= 3) & (p >= 0) & (t - src > dmin)
            & (w.gather(1, src) == y) & (doc.gather(1, pc) == doc[:, 1:]))
    q = _prev_unique(_gram_codes(w, 4))[:, (t + 1).clamp_max(length - 4)]
    qc = q.clamp_min(0)
    start = ((t + 1 <= length - 4) & (q >= 0) & (t - q > dmin)
             & ((q == 0) | (w.gather(1, (qc - 1).clamp_min(0)) != x))
             & (doc.gather(1, qc) == doc[:, 1:]) & ~cont)
    return torch.where(cont, src, torch.where(start, q, torch.full_like(q, -1)))


def retention_margin(keys, sources, *, heads, stride, block, rank, margin):
    """Raise repeated source cells above a detached causal norm cutoff."""
    b, n, width = keys.shape
    count = n // stride
    k = keys[:, :count * stride].view(b, count, stride, heads, width // heads)
    priority = k.float().norm(dim=-1).amax(dim=2).permute(0, 2, 1)
    pairs = []
    for bi in range(b):
        qpos = torch.nonzero(sources[bi] >= 0, as_tuple=False).flatten()
        if qpos.numel() == 0:
            continue
        cells = sources[bi, qpos] // stride
        eligible = (qpos // block) * block // stride
        valid = (eligible > rank) & (cells < eligible) & (cells < count)
        pairs.append((bi, torch.unique(torch.stack((eligible[valid], cells[valid]), dim=-1), dim=0)))
    terms = []
    for bi, rows in pairs:
        if rows.numel() == 0:
            continue
        eligible, cells = rows.unbind(-1)
        with torch.no_grad():
            scores = priority[bi].detach()[:, None, :].expand(-1, len(rows), -1)
            scores = scores.masked_fill(torch.arange(count, device=keys.device)[None, None] >= eligible[None, :, None],
                                       -torch.inf)
            cutoff = scores.topk(rank, dim=-1).values[..., -1].clamp_min(1e-6)
        source = priority[bi, :, cells]
        terms.append(F.relu(1 + margin - source / cutoff).flatten())
    if not terms:
        return keys.sum() * 0, 0
    values = torch.cat(terms)
    return values.mean(), values.numel()


class RetentionObjective:
    def __init__(self, model, *, rank=16, margin=0.02):
        self.rank, self.margin = rank, margin
        self.sources = None
        self.terms = []
        self.counts = []
        self.handles = []
        for module in model.modules():
            if type(module).__name__ != 'FSAttn':
                continue
            if module.fsb_enabled or module._is_dense_lr():
                raise ValueError('Retention supervision requires native pruned FS-L')
            self.handles.append(module.W_k.register_forward_hook(self._hook(module)))
        if not self.handles:
            raise ValueError('Retention supervision found no FS-L layers')

    def _hook(self, attn):
        def capture(module, inputs, output):
            if self.sources is not None:
                loss, count = retention_margin(output, self.sources, heads=attn.n_heads,
                    stride=attn.cell_stride, block=attn._cell_block, rank=self.rank, margin=self.margin)
                self.terms.append(loss)
                self.counts.append(count)
        return capture

    def begin(self, x, y):
        self.sources = repeat_sources(x, y)
        self.terms.clear()
        self.counts.clear()

    def finish(self):
        self.sources = None
        if len(self.terms) != len(self.handles):
            raise RuntimeError('Retention hooks must execute once per layer')
        loss = torch.stack(self.terms).mean()
        count = sum(self.counts)
        self.terms.clear()
        self.counts.clear()
        return loss, count
