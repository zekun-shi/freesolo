import argparse
import hashlib
import json
from pathlib import Path
import random
import tempfile
import urllib.request
import zipfile

TASKS = ["multifieldqa_en", "hotpotqa", "2wikimqa", "repobench-p"]
SIZES = [150, 300, 300, 300]


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "FreeSolo-research-evaluation"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def pilot_indices(records, n=50):
    bins = [[], [], []]
    for i, r in enumerate(records):
        b = 0 if r["length"] < 4000 else 1 if r["length"] < 8000 else 2
        bins[b].append(i)
    rng = random.Random(20260928)
    for group in bins:
        rng.shuffle(group)
    selected = []
    while len(selected) < n:
        for group in bins:
            if group and len(selected) < n:
                selected.append(group.pop())
        if not any(bins) and len(selected) < n:
            raise ValueError("Not enough pilot records")
    return sorted(selected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    hf = json.loads(fetch("https://huggingface.co/api/datasets/THUDM/LongBench"))
    gh = json.loads(fetch("https://api.github.com/repos/THUDM/LongBench/commits/main"))
    manifest = {"dataset": "THUDM/LongBench", "dataset_revision": hf["sha"],
                "protocol_revision": gh["sha"], "files": {}, "pilot_seed": 20260928}
    protocol = out / "protocol"
    protocol.mkdir()
    for name in ["config/dataset2prompt.json", "config/dataset2maxlen.json",
                 "metrics.py", "eval.py", "pred.py"]:
        data = fetch(f"https://raw.githubusercontent.com/THUDM/LongBench/{gh['sha']}/LongBench/{name}")
        p = protocol / Path(name).name
        p.write_bytes(data)
        manifest["files"][str(p.relative_to(out))] = hashlib.sha256(data).hexdigest()
    url = f"https://huggingface.co/datasets/THUDM/LongBench/resolve/{hf['sha']}/data.zip"
    print("DOWNLOADING", url, flush=True)
    with tempfile.TemporaryDirectory(prefix="freesolo-longbench-") as temporary:
        archive = Path(temporary) / "data.zip"
        urllib.request.urlretrieve(url, archive)
        manifest["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
        with zipfile.ZipFile(archive) as z:
            for task, count in zip(TASKS, SIZES):
                name = task + "_e.jsonl"
                matches = [n for n in z.namelist() if Path(n).name == name]
                assert len(matches) == 1, (name, matches)
                data = z.read(matches[0])
                records = [json.loads(line) for line in data.splitlines()]
                assert len(records) == count, (name, len(records), count)
                (out / name).write_bytes(data)
                manifest["files"][name] = hashlib.sha256(data).hexdigest()
                manifest.setdefault("tasks", {})[task] = dict(n=len(records),
                                                              pilot_indices=pilot_indices(records))
                print("PREPARED", task, len(records), flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("LONGBENCH_DATA_READY", flush=True)


if __name__ == "__main__":
    main()
