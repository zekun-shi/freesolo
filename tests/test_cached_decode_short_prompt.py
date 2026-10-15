import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# kv_cache.py imports longctx_infer by its bare name, so put experiments/scaling on the path
# (as eval_ruler.py does).
sys.path[:0] = [os.path.join(_REPO, "experiments", "scaling"), _REPO]

from experiments.scaling.model import FreeSoloLM
from kv_cache import cached_generate

# The released attention settings (eval_ruler.py with the README flags) on a tiny random model.
# FS-L adds the bounded cell pool of scripts/eval_ruler_fsl.sh. Both flags are set explicitly so
# that FS_PRUNE_KERNEL / FS_FUSED_ATTN in the environment cannot change the path under test.
FS_KW = dict(long_range_variant="cell_topk", window_size=64, cell_select="score", cell_topk=4,
             cell_stride=32, sel_strict_causal=1, use_rope=True, fs_uniform_sr=True,
             rope_long_range=False, cell_prune_kernel=0, fs_fused_attn=0)
FSL_KW = dict(FS_KW, cell_prune_kernel=1, cell_prune_pool=64, cell_prune_window=8,
              cell_prune_blkchunk=16)
# Prompts shorter than one cell (FS-L cached decode used to raise AttributeError for <= 30
# tokens), the first lengths that complete a cell, and decoding up to TOTAL tokens, so the
# long-range branch switches on (position 128) during generation.
PROMPT_LENS = (1, 11, 30, 31)
TOTAL = 144


def _tiny(kw):
    torch.manual_seed(0)
    m = FreeSoloLM(vocab_size=256, d_model=128, n_layers=2, n_heads=4, d_ff=256,
                   max_seq_len=1024, attn_type="fsattn", **kw)
    return m.float().eval()


def _check_short_prompts(kw, tag):
    m = _tiny(kw)
    for P in PROMPT_LENS:
        torch.manual_seed(1000 + P)
        prompt = torch.randint(0, 256, (1, P))
        T = TOTAL - P
        out = cached_generate(m, prompt, T, device="cpu", dtype="float32")
        assert out.shape == (1, TOTAL), f"{tag} P={P}: output shape {tuple(out.shape)}"
        # Same tokens as uncached greedy decoding: each generated token is the argmax of the
        # uncached (reference path) logits at the previous position.
        with torch.no_grad():
            ref = m(out).argmax(dim=-1)
        assert torch.equal(out[:, P:], ref[:, P - 1:-1]), \
            f"{tag} P={P}: cached decode differs from uncached greedy"
        print(f"{tag} P={P}: cached decode of {T} tokens == uncached greedy")


def test_fs_cached_decode_short_prompt():
    _check_short_prompts(FS_KW, "FS")


def test_fsl_cached_decode_short_prompt():
    _check_short_prompts(FSL_KW, "FS-L")


if __name__ == "__main__":
    test_fs_cached_decode_short_prompt()
    test_fsl_cached_decode_short_prompt()
