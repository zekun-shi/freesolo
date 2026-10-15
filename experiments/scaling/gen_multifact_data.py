import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

ROOT = Path(__file__).resolve().parents[2]

TAIL_TOKENS = 512     # models' local window: every fact must end before the final 512 tokens
PLACE_MARGIN = 64     # placement targets this many tokens of extra headroom before the cutoff
VERIFY_MARGIN = 16    # verified (stricter-than-spec) cutoff: fact end < token N - 512 - 16
LENGTH_TOL = 0.05     # accepted tokenized prefix length: L * (1 +- 0.05)
MAX_ATTEMPTS = 64

SYLLABLES = ["bar", "del", "fen", "gor", "hul", "jat", "kel", "lom", "mer", "nir",
             "pol", "quar", "rin", "sol", "tav", "ulm", "ven", "wex", "yor", "zan"]
# Fact-key attribute pools, disjoint from the background pools below: the key phrase must
# never collide with filler prose.
KEY_ADJ = ["amber", "ashen", "cobalt", "crimson", "gilded", "hollow", "ivory", "jade",
           "mottled", "obsidian", "russet", "sable", "saffron", "umber", "verdant", "violet"]
KEY_NOUN = ["cadence", "cipher", "coefficient", "dividend", "exponent", "gradient",
            "index", "modulus", "offset", "quotient", "registry", "residue",
            "tally", "threshold", "variance", "yield"]
# Neutral synthetic background prose (seeded filler).
BG_ADJ = ["broad", "calm", "damp", "early", "faded", "gentle", "hazy", "idle", "low",
          "mild", "narrow", "open", "pale", "rough", "slow", "steep", "thin", "warm",
          "wide", "young"]
BG_NOUN = ["bridge", "canal", "cellar", "cliff", "door", "field", "forest", "garden",
           "harbor", "hill", "lane", "market", "meadow", "mill", "orchard", "path",
           "pond", "ridge", "road", "shore", "tower", "valley", "village", "wall"]
BG_VERB = ["bends", "carries", "crosses", "drifts", "faces", "follows", "guards",
           "holds", "joins", "leans", "lies", "marks", "meets", "opens", "overlooks",
           "passes", "rests", "runs", "shelters", "turns"]
BG_PREP = ["above", "across", "along", "behind", "below", "beside", "beyond", "near",
           "past", "toward"]
BG_TIME = ["at dawn", "at dusk", "in autumn", "in early spring", "in late summer",
           "in winter", "after the rains", "before the harvest", "in the morning",
           "by evening"]
VALUE_CHARS = "abcdefghjkmnpqrstuvwxyz23456789"   # no i/l/o/0/1 lookalikes

# RULER S-NIAH instruction preamble.
PREAMBLE = ("Some special magic numbers are hidden within the following text. Make sure to "
            "memorize it. I will quiz you about the numbers afterwards.\n\n")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_describe():
    try:
        r = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:
        return None


def make_root(rng, n):
    return "".join(rng.choice(SYLLABLES) for _ in range(n)).capitalize()


def make_entities(rng, count, similarity):
    entities, seen = [], set()
    if similarity == "distinct":
        while len(entities) < count:
            e = make_root(rng, 3)
            if e not in seen:
                seen.add(e)
                entities.append(e)
    elif similarity == "subword":
        root = make_root(rng, 2)
        while len(entities) < count:
            e = root + "".join(rng.choice(SYLLABLES) for _ in range(2))
            if e not in seen:
                seen.add(e)
                entities.append(e)
    else:  # near: identical except the one distinguishing tag field
        root = make_root(rng, 2)
        for tag in rng.sample(range(11, 990), count):
            entities.append(f"{root} station {tag}")
    return entities


def make_value(rng, taken):
    # RULER-style 7-digit numeric value; digits never occur in the letters-only background
    # prose, so uniqueness/no-collision holds.
    while True:
        v = str(rng.randint(1000000, 9999999))
        if v in taken:
            continue
        taken.add(v)
        return v


def bg_sentence(rng):
    a, a2 = rng.choice(BG_ADJ), rng.choice(BG_ADJ)
    n, n2 = rng.choice(BG_NOUN), rng.choice(BG_NOUN)
    v, p, t = rng.choice(BG_VERB), rng.choice(BG_PREP), rng.choice(BG_TIME)
    kind = rng.randrange(4)
    if kind == 0:
        return f"The {a} {n} {v} {p} the {a2} {n2}."
    if kind == 1:
        return f"The {a} {n} {v} {p} the {n2} {t}."
    if kind == 2:
        return f"{t.capitalize()}, the {a} {n} {v} {p} the {a2} {n2}."
    return f"The {n} {p} the {a2} {n2} {v} {t}."


def assemble(units):
    # Join sentences into prose; a paragraph break every 5th unit (deterministic).
    parts = []
    for i, u in enumerate(units):
        if i:
            parts.append("\n\n" if i % 5 == 0 else " ")
        parts.append(u)
    return "".join(parts)


def ntokens(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def build_record(rng, tokenizer, a, index):
    F = a.facts
    if a.similarity == "distinct":
        # A unique (adjective, noun) attribute per fact in addition to a unique entity;
        # shared-attribute keys are only used at the subword/near similarity levels.
        attrs = rng.sample([(x, y) for x in KEY_ADJ for y in KEY_NOUN], F)
    else:
        adj0, noun0 = rng.choice(KEY_ADJ), rng.choice(KEY_NOUN)
        attrs = [(adj0, noun0)] * F
    entities = make_entities(rng, F, a.similarity)
    taken: set = set()
    values = [make_value(rng, taken) for _ in range(F)]
    keys = [f"{attrs[i][0]} {attrs[i][1]} of {entities[i]}" for i in range(F)]
    assert len(set(keys)) == F and len(set(values)) == F, "duplicate keys/values"

    def fact_sentence(i):
        # RULER S-NIAH needle template; the key is "<adj> <noun> of <entity>" so the
        # distinct/subword/near machinery holds.
        return (f"One of the special magic numbers for the {attrs[i][0]} {attrs[i][1]} "
                f"of {entities[i]} is: {values[i]}.")

    sentences = [fact_sentence(i) for i in range(F)]
    fact_tok = sum(ntokens(tokenizer, s) for s in sentences)
    hard_tail = TAIL_TOKENS + PLACE_MARGIN
    assert fact_tok <= a.length - hard_tail - 128, (
        f"{F} facts ~{fact_tok} tokens cannot sit before the last-{TAIL_TOKENS} cutoff "
        f"at L={a.length}")

    bg, per = [], []              # background sentences + their token counts
    while sum(per) + fact_tok < a.length:
        s = bg_sentence(rng)
        bg.append(s)
        per.append(ntokens(tokenizer, s))

    lo, hi = int(a.length * (1 - LENGTH_TOL)), int(a.length * (1 + LENGTH_TOL))
    for attempt in range(1, MAX_ATTEMPTS + 1):
        # Eligible insertion slots (before bg sentence j): worst case ALL facts share the
        # latest slot, so its cumulative background position must leave fact_tok + hard_tail
        # tokens after it.
        total_est = sum(per) + fact_tok
        budget = total_est - hard_tail - fact_tok
        cum, eligible = 0, [0]
        for j, t in enumerate(per):
            cum += t
            if cum <= budget:
                eligible.append(j + 1)
        assert eligible, "no eligible fact slot; increase --length or reduce --facts"
        slots = sorted(rng.choices(eligible, k=F))   # random positions, may share a slot

        units, fi = [], 0
        for j in range(len(bg) + 1):
            while fi < F and slots[fi] == j:
                units.append(sentences[fi])
                fi += 1
            if j < len(bg):
                units.append(bg[j])
        assert fi == F
        # RULER-style instruction preamble (needle/continuation framing, not QA framing).
        text = PREAMBLE + assemble(units)
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        N = len(enc["input_ids"])
        if N > hi:                                   # trim background from the fact-free tail
            drop = max(1, (N - a.length) // max(1, per[-1]))
            del bg[-drop:], per[-drop:]
            continue
        if N < lo:                                   # pad background at the end
            for _ in range(max(1, (a.length - N) // 12)):
                s = bg_sentence(rng)
                bg.append(s)
                per.append(ntokens(tokenizer, s))
            continue
        # char position where the final (512 + VERIFY_MARGIN) tokens begin
        cut = enc["offset_mapping"][N - TAIL_TOKENS - VERIFY_MARGIN][0]
        low = text.lower()
        ok, facts = True, []
        for i in range(F):
            fs = sentences[i]
            p = text.index(fs)
            if p + len(fs) > cut:                    # fact leaked into the local window
                ok = False
                break
            if low.count(values[i]) != 1:            # accidental occurrence: new value
                values[i] = make_value(rng, taken)
                sentences[i] = fact_sentence(i)
                ok = False
                break
            facts.append(dict(key_desc=keys[i], entity=entities[i], value=values[i],
                              char_pos=p))
        if not ok:
            continue
        Q = min(a.questions_per_prefix or min(F, 8), F)
        qidx = sorted(rng.sample(range(F), Q))
        # RULER-style answer-prefix continuation.
        questions = [dict(q=(f"\nThe special magic number for the {attrs[i][0]} {attrs[i][1]} "
                             f"of {entities[i]} mentioned in the context is"),
                          answer=values[i], fact_idx=i) for i in qidx]
        return dict(prefix=text, facts=facts, questions=questions, n_tokens=N), attempt
    raise RuntimeError(f"prefix {index}: no valid layout after {MAX_ATTEMPTS} attempts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-prefixes", type=int, default=200)
    ap.add_argument("--facts", type=int, default=4)
    ap.add_argument("--length", type=int, default=4096)      # target tokens, +-5%
    ap.add_argument("--similarity", choices=["distinct", "subword", "near"], default="distinct")
    ap.add_argument("--questions-per-prefix", type=int, default=0)   # 0 -> min(facts, 8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tokenizer", default="fla-hub/transformer-1.3B-100B")
    a = ap.parse_args()
    assert a.facts >= 1 and a.num_prefixes >= 1
    assert a.length > 2 * (TAIL_TOKENS + PLACE_MARGIN), "length must clear the local-window tail"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer   # length calibration only; generation is text-based
    local_only = os.environ.get("HF_HUB_OFFLINE", "0") not in ("", "0")
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=local_only)
    assert tokenizer.is_fast

    name = f"cap_{a.similarity}_f{a.facts}_L{a.length}.jsonl"
    lengths, attempts = [], []
    with (out / name).open("w") as stream:
        for i in range(a.num_prefixes):
            rng = random.Random(f"{a.seed}:{a.similarity}:{a.facts}:{a.length}:{i}")
            rec, tries = build_record(rng, tokenizer, a, i)
            stream.write(json.dumps(rec) + "\n")
            lengths.append(rec["n_tokens"])
            attempts.append(tries)
            if (i + 1) % 20 == 0 or i + 1 == a.num_prefixes:
                print(f"DONE {i + 1}/{a.num_prefixes} tokens={rec['n_tokens']}", flush=True)

    manifest = dict(args=vars(a), seed=a.seed, git_describe=git_describe(),
                    tail_tokens=TAIL_TOKENS, verify_margin=VERIFY_MARGIN,
                    files={name: digest(out / name)},
                    stats=dict(n_prefixes=a.num_prefixes, min_tokens=min(lengths),
                               max_tokens=max(lengths),
                               mean_tokens=sum(lengths) / len(lengths),
                               max_attempts=max(attempts)))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("COMPLETE", out / name, flush=True)


if __name__ == "__main__":
    main()
