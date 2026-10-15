import argparse
import json
import random
from collections import defaultdict

LENGTH_BINS = ((0, 4000, "<4000"), (4000, 8000, "<8000"), (8000, None, ">=8000"))


def iter_rows(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_run(path):
    manifest, examples = None, []
    for row in iter_rows(path):
        kind = row.get("kind")
        if kind == "manifest":
            manifest = row
        elif kind == "example":
            examples.append(row)
    if manifest is None or not examples:
        raise ValueError(f"{path}: expected an eval_longbench.py output with manifest and example rows")
    args = manifest.get("args", {})
    if not args.get("task"):
        raise ValueError(f"{path}: manifest has no task")
    return dict(path=path, task=args["task"], args=args, examples=examples)


def run_name(run):
    return run["label"] + ("-nocontext" if run["without_context"] else "")


def percentile(values, p):
    s = sorted(values)
    pos = (len(s) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def mean100(rows):
    return 100.0 * sum(r["score"] for r in rows) / len(rows)


def clusters(rows):
    # resample unit: the document context cluster (examples sharing document_sha256);
    # sorted for a deterministic draw order
    groups = defaultdict(list)
    for r in rows:
        groups[r["document_sha256"]].append(r)
    return [g for _, g in sorted(groups.items(), key=lambda kv: min(r["index"] for r in kv[1]))]


def bootstrap_means(groups, resamples, rng):
    k = len(groups)
    out = []
    for _ in range(resamples):
        tot = n = 0.0
        for _ in range(k):
            g = groups[rng.randrange(k)]
            tot += sum(r["score"] for r in g)
            n += len(g)
        out.append(100.0 * tot / n)
    return out


def paired_intervals(run_a, run_b, resamples, rng):
    # paired cluster bootstrap over the shared example indices: one cluster draw applied to
    # both runs gives a resample of each run's mean and of the a-minus-b difference
    by_a = {r["index"]: r for r in run_a["examples"]}
    by_b = {r["index"]: r for r in run_b["examples"]}
    common = sorted(set(by_a) & set(by_b))
    if not common:
        return None
    pairs = defaultdict(list)
    for i in common:
        ra, rb = by_a[i], by_b[i]
        assert ra["document_sha256"] == rb["document_sha256"]
        pairs[ra["document_sha256"]].append((ra, rb))
    groups = [v for _, v in sorted(pairs.items(), key=lambda kv: min(p[0]["index"] for p in kv[1]))]
    k = len(groups)
    ma, mb, md = [], [], []
    for _ in range(resamples):
        ta = tb = n = 0.0
        for _ in range(k):
            g = groups[rng.randrange(k)]
            ta += sum(p[0]["score"] for p in g)
            tb += sum(p[1]["score"] for p in g)
            n += len(g)
        va, vb = 100.0 * ta / n, 100.0 * tb / n
        ma.append(va)
        mb.append(vb)
        md.append(va - vb)
    point_a = mean100([by_a[i] for i in common])
    point_b = mean100([by_b[i] for i in common])
    return dict(n_common=len(common), clusters=k,
                mean_a=point_a, ci_a=[percentile(ma, 0.025), percentile(ma, 0.975)],
                mean_b=point_b, ci_b=[percentile(mb, 0.025), percentile(mb, 0.975)],
                diff=point_a - point_b, ci_diff=[percentile(md, 0.025), percentile(md, 0.975)])


def strata(run):
    bins = {name: [] for _, _, name in LENGTH_BINS}
    for r in run["examples"]:
        length = r["benchmark_length"]
        for lo, hi, name in LENGTH_BINS:
            if length >= lo and (hi is None or length < hi):
                bins[name].append(r)
                break
    return bins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="eval_longbench.py example-output .jsonl files")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="one model label per --inputs file (default: the manifest architecture)")
    ap.add_argument("--seed", type=int, default=20260928, help="bootstrap rng seed")
    ap.add_argument("--resamples", type=int, default=5000, help="paired bootstrap resamples")
    ap.add_argument("--strata", action="store_true",
                    help="also report scores by benchmark length bin (<4000, <8000, >=8000 words)")
    ap.add_argument("--output", default=None, help="optional json dump of the full report")
    a = ap.parse_args()
    if a.labels is not None and len(a.labels) != len(a.inputs):
        ap.error("--labels must give one label per --inputs file")

    runs = []
    for i, path in enumerate(a.inputs):
        run = load_run(path)
        run["without_context"] = bool(run["args"].get("without_context"))
        run["label"] = a.labels[i] if a.labels else str(run["args"].get("architecture"))
        runs.append(run)
    seen = defaultdict(int)
    for run in runs:
        seen[(run["task"], run_name(run))] += 1
    dup = [k for k, v in seen.items() if v > 1]
    if dup:
        ap.error(f"ambiguous runs (same task and label): {dup}; pass --labels")

    by_task = defaultdict(list)
    for run in runs:
        by_task[run["task"]].append(run)

    report = {"seed": a.seed, "resamples": a.resamples, "tasks": {}}
    for task in sorted(by_task):
        truns = sorted(by_task[task], key=lambda r: (r["label"], r["without_context"]))
        section = {"runs": {}, "differences": {}, "with_minus_without": {}}
        report["tasks"][task] = section
        print(f"== {task} ==")
        for run in truns:
            name = run_name(run)
            groups = clusters(run["examples"])
            rng = random.Random(f"{a.seed}:{task}:{name}")
            boots = bootstrap_means(groups, a.resamples, rng)
            mean = mean100(run["examples"])
            lo, hi = percentile(boots, 0.025), percentile(boots, 0.975)
            entry = dict(n=len(run["examples"]), clusters=len(groups), mean=round(mean, 4),
                         ci95=[round(lo, 4), round(hi, 4)])
            if a.strata:
                entry["strata"] = {b: dict(n=len(rows), mean=round(mean100(rows), 4))
                                   for b, rows in strata(run).items() if rows}
            section["runs"][name] = entry
            print(f"  {name:<24} mean {mean:6.2f}  95% CI [{lo:6.2f}, {hi:6.2f}]"
                  f"  (n={entry['n']}, clusters={entry['clusters']})")
            for b, st in entry.get("strata", {}).items():
                print(f"      {b:<10} n={st['n']:<4d} mean {st['mean']:6.2f}")
        for i in range(len(truns)):
            for j in range(i + 1, len(truns)):
                r1, r2 = truns[i], truns[j]
                same_model = r1["label"] == r2["label"]
                context_pair = same_model and r1["without_context"] != r2["without_context"]
                model_pair = not same_model and not r1["without_context"] and not r2["without_context"]
                if not (context_pair or model_pair):
                    continue
                with_run, other_run = (r2, r1) if r1["without_context"] else (r1, r2)
                rng = random.Random(f"{a.seed}:{task}:{run_name(with_run)}:{run_name(other_run)}")
                res = paired_intervals(with_run, other_run, a.resamples, rng)
                key = f"{run_name(with_run)} - {run_name(other_run)}"
                if res is None:
                    print(f"  diff {key}: no shared example indices; skipped")
                    continue
                entry = dict(n_common=res["n_common"], clusters=res["clusters"],
                             diff=round(res["diff"], 4),
                             ci95=[round(res["ci_diff"][0], 4), round(res["ci_diff"][1], 4)])
                if context_pair:
                    section["with_minus_without"][with_run["label"]] = entry
                else:
                    section["differences"][key] = dict(
                        entry, means={run_name(with_run): round(res["mean_a"], 4),
                                      run_name(other_run): round(res["mean_b"], 4)})
                print(f"  diff {key:<38} {res['diff']:+6.2f}  95% CI [{res['ci_diff'][0]:+6.2f},"
                      f" {res['ci_diff'][1]:+6.2f}]  (n={res['n_common']}, clusters={res['clusters']})")
    if a.output:
        with open(a.output, "w") as f:
            json.dump(report, f, indent=2)
        print("WROTE", a.output)


if __name__ == "__main__":
    main()
