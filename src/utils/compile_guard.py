"""torch.compile guard of the hand-written inference paths.

While TorchDynamo traces a function (torch.compile, torch.export), compiling() is True, and every hand-written Triton
or ctypes CUDA inference path (trunk_triton, readout_triton, the batched window, the FS-L and FS-B CUDA kernels, the
fast FS-B prefill and the glue that hands their results around) turns itself off. Dynamo would otherwise trace them:
torch 2.5 passes Triton's launch options (enable_fp_fusion) as kernel arguments, which gave NaN and wrong logits, and
every ctypes launch is a graph break. A compiled model therefore runs the eager reference ops, which inductor compiles
as it did before these paths existed. Outside a trace compiling() is False and costs one flag read."""
import torch

_is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", None)
if _is_compiling is None:  # pragma: no cover  (torch < 2.3)
    try:
        from torch._dynamo import is_compiling as _is_compiling
    except ImportError:
        def _is_compiling() -> bool:
            return False


def compiling() -> bool:
    """True while TorchDynamo traces the caller; dynamo folds the call to a constant."""
    return _is_compiling()
