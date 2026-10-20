"""Exercise the report/log plumbing without loading a checkpoint or CUDA kernels."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


FAST_FLAGS = ("FSB_FAST_BEAM", "FSB_FAST_RERANK", "FSB_FAST_CACHE",
              "FSB_FAST_BUILD", "FSB_FAST_ROWS", "FSB_DECODE_WHOLE", "FSB_BEAM_CUDA",
              "FSB_BEAM_ROWS", "FSB_TRAVERSE_TRITON", "FSB_BUILD_CUDA")
GRAPH_FLAGS = ("FSB_DECODE_GRAPH", "FSB_DECODE_GRAPH_CHECK", "FSB_TRAIN_GRAPH")
ROWS_FLAGS = {"FSB_ROWS_IMPL": "cuda", "FSB_ROWS_DEFER": "1", "FSB_ROWS_LOSKIP": "1", "FSB_ROWS_LSE_SPLIT": "4",
              "FSB_BEAM_COMPACT": "1", "FSB_CHECK_COMPACT": "0", "FSB_FAST_PREFILL": "1",
              "FSB_FAST_PREFILL_BLKCHUNK": "128", "FSB_FAST_PREFILL_SCRATCH_MB": "2048", "FSB_PREFILL_STATS": "0",
              "FSB_WINDOW_OVERLAP": "1", "FSB_BUILD_PREFILL_TRIM": "1", "FSB_PREFILL_NOCOPY": "1", "FSB_ROWS_V3": "1",
              "FSB_BEAM_V2": "1", "FSB_BEAM_V2_EXACT": "1"}


@pytest.fixture
def ruler_driver(monkeypatch):
    class Tensor:
        def numel(self):
            return 3

        def __getitem__(self, key):
            return [42]

    tokenizer = SimpleNamespace(encode=lambda prompt: [1, 2, 3], decode=lambda ids: "answer",
                                eos_token_id=2, vocab_size=32000)
    torch = ModuleType("torch")
    torch.tensor = lambda *args, **kwargs: Tensor()
    torch.long = "long"
    torch.cuda = SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None,
                                 OutOfMemoryError=type("FakeOOM", (Exception,), {}))
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer)
    infer = ModuleType("longctx_infer")
    infer.MODEL_CONFIGS = {"340m": {}}
    infer.load_model = lambda *args, **kwargs: SimpleNamespace(modules=lambda: [])
    cache = ModuleType("kv_cache")
    cache.cached_generate = lambda *args, **kwargs: Tensor()
    tree = ModuleType("src.fsattn.fsb_tree")
    tree.fsb_read_stats_reset = lambda model: None
    tree.fsb_read_stats = lambda model: dict(mean=0, rows=0, max=0, layer_mean=[])
    route = ModuleType("src.fsattn.fsb_route")
    route.router_key_check = lambda *args, **kwargs: (None, set())
    for module in (torch, transformers, infer, cache, tree, route):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    for name in (*FAST_FLAGS, *GRAPH_FLAGS, *ROWS_FLAGS, "FSB_DECODE_V2", "FSB_DECODE_FORK", "FS_PRUNE_KERNEL",
                 "FS_FUSED_ATTN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    path = Path(__file__).resolve().parents[1] / "eval_ruler.py"
    spec = importlib.util.spec_from_file_location("ruler_metadata_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("overrides", [
    {},
    {**{name: "0" for name in FAST_FLAGS}, "FSB_DECODE_GRAPH": "0"},
    {"FSB_DECODE_GRAPH": "0"},
    {"FSB_DECODE_GRAPH_CHECK": "1"},
    {"FSB_DECODE_V2": "0"},
    {"FSB_DECODE_V2": "0", "FSB_DECODE_FORK": "0"},
    {"FSB_BEAM_CUDA": "0"},
    {"FSB_BEAM_ROWS": "0"},
    {"FSB_TRAVERSE_TRITON": "0"},
    {"FSB_BUILD_CUDA": "0"},
    {"FSB_FAST_BUILD": "0", "FSB_FAST_ROWS": "0", "FSB_TRAIN_GRAPH": "1"},
    {"FSB_ROWS_IMPL": "legacy", "FSB_ROWS_DEFER": "0", "FSB_ROWS_LOSKIP": "0", "FSB_ROWS_LSE_SPLIT": "1",
     "FSB_BEAM_COMPACT": "0", "FSB_CHECK_COMPACT": "1"},
    {"FSB_FAST_PREFILL": "0", "FSB_FAST_PREFILL_BLKCHUNK": "32", "FSB_FAST_PREFILL_SCRATCH_MB": "512",
     "FSB_PREFILL_STATS": "1"},
    {"FSB_WINDOW_OVERLAP": "0"},
    {"FSB_BUILD_PREFILL_TRIM": "0", "FSB_PREFILL_NOCOPY": "0", "FSB_ROWS_V3": "0", "FSB_BEAM_V2": "0",
     "FSB_BEAM_V2_EXACT": "0"},
    {"FSB_BEAM_V2_EXACT": "0"},
])
def test_fsb_report_and_log_preserve_resolved_flags(ruler_driver, monkeypatch, tmp_path, capsys, overrides):
    for name, value in overrides.items():
        monkeypatch.setenv(name, value)
    task = tmp_path / "niah_single_1"
    task.mkdir()
    (task / "validation.jsonl").write_text(json.dumps({"input": "question", "outputs": ["answer"]}) + "\n")
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["eval_ruler.py", "--fsb", "--ckpt", str(tmp_path / "model"),
                                     "--data_dir", str(tmp_path), "--tasks", "niah_single_1",
                                     "--length", "2048", "--limit", "1", "--output", str(output)])
    ruler_driver.main()
    report = json.loads(output.read_text())
    assert report["results"]["niah_single_1"]["score"] == 100
    assert report["results"]["niah_single_1"]["n"] == 1
    flags = report["meta"]["fsb_execution_env"]
    expected = {**dict.fromkeys(FAST_FLAGS, "1"), "FSB_DECODE_GRAPH": "1",
                "FSB_DECODE_GRAPH_CHECK": "0", "FSB_DECODE_V2": "1", "FSB_DECODE_FORK": "1", "FSB_TRAIN_GRAPH": "0",
                **ROWS_FLAGS, **overrides}
    assert flags == expected
    env_line = next(line for line in capsys.readouterr().out.splitlines() if "FS-B eval env:" in line)
    for name, value in expected.items():
        assert f"{name}={value}" in env_line.split()


def test_bench_rows_record_the_ruler_fsb_switch_table(ruler_driver, monkeypatch):
    # bench_efficiency.py's FS-B rows record the same switch table (read from its source: the fixture fakes torch)
    import ast
    src = (Path(__file__).resolve().parents[1] / "experiments" / "scaling" / "bench_efficiency.py").read_text()
    table = next(ast.literal_eval(node.value) for node in ast.parse(src).body if isinstance(node, ast.Assign)
                 and any(getattr(t, "id", None) == "FSB_FLAGS" for t in node.targets))
    for name in table:
        monkeypatch.delenv(name, raising=False)
    assert ruler_driver._fsb_execution_env() == table


ATTN_DEFAULTS = {"FSL_SLIDING_MODE": "chunked", "FSL_WINDOW_BATCHED": "1", "FSL_TRUNK_FUSED": "1",
                 "FSL_NORM_ONEPASS": "1", "FSL_ROPE_FUSED": "1", "FSL_PREFILL_CELL_REUSE": "1",
                 "FSL_READOUT_FUSED": "1", "FSL_WINDOW_STREAM": "0", "FSL_READOUT_G": "64", "FSL_READOUT_WARPS": "4",
                 "FSL_READOUT_STAGES": "2", "FSL_FAST_SELECT": "1", "FSL_FUSED_CONTENT": "1",
                 "FSL_FAST_CACHE_STATS": "1", "FS_WINDOW_OVERLAP": "1", "FS_DECODE_FORK": "1", "FSL_KV_BOUNDED": "1"}
ATTN_FLAGS = tuple(k for k in ATTN_DEFAULTS if k != "FSL_SLIDING_MODE")


@pytest.mark.parametrize("fsb", [True, False])
@pytest.mark.parametrize("overrides", [{}, {name: "0" for name in ATTN_FLAGS}, {"FSL_SLIDING_MODE": "off"},
                                       {"FSL_WINDOW_STREAM": "1"}, {"FS_WINDOW_OVERLAP": "0", "FS_DECODE_FORK": "0"},
                                       {"FSL_KV_BOUNDED": "0"}])
def test_attention_execution_env_recorded(ruler_driver, monkeypatch, tmp_path, capsys, fsb, overrides):
    for name in (*ATTN_FLAGS, "FSL_SLIDING_MODE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in overrides.items():
        monkeypatch.setenv(name, value)
    task = tmp_path / "niah_single_1"
    task.mkdir()
    (task / "validation.jsonl").write_text(json.dumps({"input": "question", "outputs": ["answer"]}) + "\n")
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["eval_ruler.py", *(["--fsb"] if fsb else []), "--ckpt", str(tmp_path / "model"),
                                     "--data_dir", str(tmp_path), "--tasks", "niah_single_1",
                                     "--length", "2048", "--limit", "1", "--output", str(output)])
    ruler_driver.main()
    flags = json.loads(output.read_text())["meta"]["attn_execution_env"]
    expected = {**ATTN_DEFAULTS, **overrides}
    assert flags == expected
    env_line = next(line for line in capsys.readouterr().out.splitlines() if "attention eval env:" in line)
    for name, value in expected.items():
        assert f"{name}={value}" in env_line.split()
