import json
import os
import subprocess
from pathlib import Path

import pytest

from experiments.scaling.eval_report import result_complete

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def saved_result(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text('{}\n{}\n')
    path = tmp_path / "result.json"
    args = {"task": "niah_single_1", "length": 4096, "checkpoint": str(tmp_path / "model"),
            "variant": "FS", "data": data, "limit": 2}
    report = {"meta": {"length": 4096, "ckpt": args["checkpoint"], "variant": "FS"},
              "results": {args["task"]: {"status": "ok", "n": 2, "score": 100.0}}}
    path.write_text(json.dumps(report))
    return path, args, report


def test_complete_result_is_reusable(saved_result):
    path, args, _ = saved_result
    assert result_complete(path, **args)


@pytest.mark.parametrize("failure", ["error", "partial", "corrupt", "variant", "checkpoint", "length", "nan"])
def test_incomplete_or_different_result_is_not_reused(saved_result, failure):
    path, args, report = saved_result
    result = report["results"][args["task"]]
    if failure == "error":
        result["status"] = "OOM"
    elif failure == "partial":
        result["n"] = 1
    elif failure == "variant":
        report["meta"]["variant"] = "FS-L"
    elif failure == "checkpoint":
        report["meta"]["ckpt"] = "another-model"
    elif failure == "length":
        report["meta"]["length"] = 2048
    elif failure == "nan":
        result["score"] = float("nan")
    path.write_text("{" if failure == "corrupt" else json.dumps(report))
    assert not result_complete(path, **args)


@pytest.mark.parametrize("script", ["eval_ruler_fs.sh", "eval_ruler_fsl.sh"])
@pytest.mark.parametrize("mode", ["process-error", "report-error", "ok"])
def test_grid_reports_failure_and_only_reuses_complete_jobs(tmp_path, script, mode):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    interpreter = bin_dir / "python"
    interpreter.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
if not sys.argv[1].endswith("/eval_ruler.py"):
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
args = sys.argv[2:]
def value(flag):
    return args[args.index(flag) + 1]
with open(os.environ["CALLS"], "a") as stream:
    stream.write("run\\n")
if os.environ["FAKE_MODE"] == "process-error":
    sys.exit(7)
variant = "FS-L" if os.environ.get("FS_PRUNE_KERNEL") == "1" else "FS"
report = {"meta": {"ckpt": value("--ckpt"), "length": int(value("--length")), "variant": variant},
          "results": {value("--tasks"): {"status": "ok" if os.environ["FAKE_MODE"] == "ok" else "OOM",
                                         "score": 100, "n": 1}}}
pathlib.Path(value("--output")).write_text(json.dumps(report))
''')
    interpreter.chmod(0o755)
    data = tmp_path / "data" / "L4096" / "niah_single_1"
    data.mkdir(parents=True)
    (data / "validation.jsonl").write_text('{}\n')
    calls = tmp_path / "calls"
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
               CKPT=str(tmp_path / "checkpoint"), GPUS="0", LIMIT="1", LENS="4096",
               TASKS="niah_single_1", RULER_DATA=str(tmp_path / "data"), OUT=str(tmp_path / "out"),
               CALLS=str(calls), FAKE_MODE=mode)
    command = ["bash", str(ROOT / "scripts" / script)]
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=20, check=False)
    assert result.returncode == (0 if mode == "ok" else 1), result.stdout + result.stderr
    assert ("FULLRULER_DONE" in result.stdout) == (mode == "ok")
    if mode == "ok":
        second = subprocess.run(command, env=env, text=True, capture_output=True, timeout=20, check=False)
        assert second.returncode == 0
        assert "HAVE complete" in second.stdout
        assert calls.read_text() == "run\n"
