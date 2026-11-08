"""Final SWAN743 on the retained Table 18 shared-prefix questions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import time

ROOT = Path('/home/aiops/shizk/freesolo_rebuttal_20261107/gb200')
SOURCE = ROOT / 'code/b879ec5c8fe8'
CHECKPOINT = ROOT / 'runs/swan743_32b_s42_final/swan743_32b_s42_final_step61035'
OUT = ROOT / 'multifact_swan743_final_v1'
WEIGHT = '415c663c772593e46f1c280c85b9a95f664b9d3db5c0825e4717a0c6f3596304'
COMMIT = 'b879ec5c8fe8b0b2037fadd49dbb2933d3bbfc29'
MANIFEST = 'ae3ff66689115bf3e58e7a542479e566fc5c22f064bbb7db73f45ecde17b375e'
DATA = {
    1: '568efdbfe02aa847a623919b6b9ebca3deeb6a2e0b541d1756809f32acf44009',
    2: '7124bfdcf9e702bc6213b02933d86fda6cd2e8a2d48b07128793bcee4196966c',
    4: 'e267d800dde093e917d64a4a9d14add0140cf11c8387fb8619b3a2b09f60a687',
    8: '3fd59d558e23c2d1185f757f7c51bd1150646b59eb46ba16295645cde95c105c',
    16: 'a3ffee3118b9919a174065245d447cf83ef62dbeb41e6a9e0dc50ca8ee65f146',
}
TOK = 'fla-hub/transformer-1.3B-100B'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def put(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def verify_source():
    assert (SOURCE / 'SOURCE_COMMIT').read_text().strip() == COMMIT
    assert sha(SOURCE / 'SOURCE_MANIFEST.json') == MANIFEST
    for name, digest in json.loads((SOURCE / 'SOURCE_MANIFEST.json').read_text()).items():
        assert sha(SOURCE / name) == digest, name


def worker(facts):
    verify_source()
    assert sha(CHECKPOINT / 'model.safetensors') == WEIGHT
    config = json.loads((CHECKPOINT / 'run_config.json').read_text())
    assert config['step'] == 61035 and config['tokens'] == 31999918080
    assert config['attn_type'] == 'swan'
    kwargs = dict(window_size=512, swan_rope_theta=1e6, swan_scale_base=8192.)
    assert config['attn_kwargs'] == kwargs
    data = ROOT / f'multifact/data/f{facts}.jsonl'
    prior = ROOT / f'multifact/prior/fsb_743_f{facts}.jsonl'
    assert sha(data) == DATA[facts]
    records = [json.loads(line) for line in data.read_text().splitlines()]
    refs = [json.loads(line) for line in prior.read_text().splitlines()]
    assert refs[0]['data_sha256'] == DATA[facts]
    refs = [r for r in refs if r['kind'] == 'example']
    assert len(records) == 50 and len(refs) == 50 * min(facts, 8)
    sys.path[:0] = [str(SOURCE), str(SOURCE / 'experiments/scaling')]
    import torch
    from transformers import AutoTokenizer
    from safetensors import safe_open
    from experiments.scaling import longctx_infer as lci
    from experiments.scaling.baseline_cache import baseline_cached_generate

    torch.set_grad_enabled(False)
    torch.set_num_threads(2)
    torch.backends.cuda.enable_cudnn_sdp(False)
    assert torch.cuda.device_count() == 1
    tok = AutoTokenizer.from_pretrained(TOK, trust_remote_code=True, local_files_only=True)
    eos = tok.eos_token_id if tok.eos_token_id is not None else 2
    lci.MODEL_CONFIGS['800m']['vocab_size'] = 32000
    model = lci.load_model(str(CHECKPOINT), config='800m', attn_type='swan',
                           max_seq_len=4096, device='cuda', **kwargs).eval()
    with safe_open(str(CHECKPOINT / 'model.safetensors'), framework='pt', device='cpu') as f:
        keys = set(f.keys())
    missing = set(model.state_dict()) - keys
    assert missing <= {'lm_head.weight'} and not keys - set(model.state_dict())
    assert model.lm_head.weight is model.tok_emb.weight
    assert sum(p.numel() for p in model.parameters()) == 742860288
    assert len(model.blocks) == 24
    for i, block in enumerate(model.blocks):
        attn = block.attn.standard
        assert block.attn.attn_type == 'swan'
        assert attn.use_rope == (i % 4 != 0)
        assert attn.window_size == (512 if i % 4 else None)
        assert attn.infer_scale_base == (0 if i % 4 else 8192)
    prepared = []
    for pi, rec in enumerate(records):
        assert len(rec['facts']) == facts and len(rec['questions']) == min(facts, 8)
        prefix = tok.encode(rec['prefix'])
        for qi, q in enumerate(rec['questions']):
            ids = prefix + tok.encode(q['q'], add_special_tokens=False)
            digest = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
            ref = refs[len(prepared)]
            identity = dict(prefix_index=pi, question_index=qi, fact_idx=q['fact_idx'],
                            gold=q['answer'], prompt_tokens=len(ids), prompt_tokens_sha256=digest)
            assert all(ref[k] == v for k, v in identity.items()), identity
            assert len(ids) + 16 <= 4096
            prepared.append((identity, ids, [f['value'] for f in rec['facts']]))
    target = OUT / f'f{facts}.jsonl'
    start = time.time()
    with target.open('x') as stream:
        def emit(row):
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
        emit(dict(kind='manifest', model='SWAN743', facts=facts, checkpoint=str(CHECKPOINT),
                  checkpoint_sha256=WEIGHT, step=61035, tokens=31999918080,
                  source_commit=COMMIT, source_manifest_sha256=MANIFEST,
                  helper_sha256=sha(Path(__file__)), data_sha256=DATA[facts],
                  reference_sha256=sha(prior), tokenizer=TOK, eos=eos, max_new=16,
                  protocol='rebuttal800', eos_stop=True, precision='bf16 autocast; fp32 LM head',
                  torch=torch.__version__, cuda=torch.version.cuda, hardware=torch.cuda.get_device_name(),
                  argv=sys.argv, attn_kwargs=kwargs, missing_keys=sorted(missing),
                  env={k:v for k,v in os.environ.items() if k.startswith(('CUDA_VISIBLE', 'HF_', 'OMP_', 'MKL_'))}))
        correct = wrong_other = 0
        for index, (identity, ids, values) in enumerate(prepared):
            began = time.time()
            prompt = torch.tensor(ids, dtype=torch.long)
            generated = baseline_cached_generate(model, prompt, 16, device='cuda',
                                                  dtype='bfloat16', eos_token_id=eos)
            tokens = generated[0, len(ids):].tolist()
            # Check the actual final checkpoint against full recomputation.
            if index == 0:
                uncached = lci.greedy_generate(model, prompt, 16, device='cuda',
                                              dtype='bfloat16', eos_token_id=eos)
                same = torch.equal(uncached, generated)
                put(OUT / f'f{facts}.cache_check.json', dict(passed=same,
                    prompt_tokens_sha256=identity['prompt_tokens_sha256'], cached=tokens,
                    full_recompute=uncached[0, len(ids):].tolist(), checkpoint_sha256=WEIGHT))
                assert same, 'cached/full-recompute token mismatch'
            pred = tok.decode(tokens, skip_special_tokens=True)
            hit = identity['gold'].lower() in pred.lower()
            other = any(v.lower() in pred.lower() for v in values if v != identity['gold'])
            correct += hit
            wrong_other += (not hit) and other
            emit(dict(kind='example', **identity, pred=pred, generated_ids=tokens, correct=hit,
                      copies_other_value=other, seconds=time.time() - began))
            if (index + 1) % 10 == 0:
                put(OUT / f'f{facts}.progress.json', dict(status='running', completed=index + 1,
                    total=len(prepared), correct=correct, at=time.time()))
        summary = dict(kind='summary', status='complete', facts=facts, n_prefixes=50,
            n_questions=len(prepared), n_correct=correct, accuracy=100 * correct / len(prepared),
            n_wrong_with_other_value=wrong_other, seconds=time.time() - start)
        emit(summary)
    put(OUT / f'f{facts}.progress.json', dict(summary, predictions_sha256=sha(target)))


def coordinate():
    OUT.mkdir(exist_ok=False)
    verify_source()
    uuids = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                    text=True).strip().splitlines()
    assert len(set(uuids)) == len(uuids) == 4
    put(OUT / 'owner.json', dict(pid=os.getpid(), host=os.uname().nodename,
        started_at=time.time(), argv=sys.argv, gpu_uuids=uuids, helper_sha256=sha(Path(__file__))))
    queue = Queue()
    for f in [16, 8, 4, 2, 1]:
        queue.put(f)
    env = {k:v for k,v in os.environ.items() if not k.startswith(('FS_', 'FSB_', 'FSL_'))}
    env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1',
        TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
        HF_HOME='/home/aiops/shizk/fsb13_diagnosis_20261105/eval/hf_home')
    def launch(gpu):
        while True:
            try:
                facts = queue.get_nowait()
            except Empty:
                return
            command = [sys.executable, str(Path(__file__).resolve()), '--facts', str(facts)]
            child_env = dict(env, CUDA_VISIBLE_DEVICES=str(gpu))
            put(OUT / f'f{facts}.command.json', dict(command=command, cwd=str(SOURCE),
                env={k:v for k,v in child_env.items() if k.startswith(('CUDA_VISIBLE', 'HF_', 'OMP_', 'MKL_'))}))
            with (OUT / f'f{facts}.log').open('x') as log:
                run = subprocess.run(command, cwd=SOURCE, env=child_env, stdout=log,
                                     stderr=subprocess.STDOUT, timeout=7200)
            put(OUT / f'f{facts}.exit.json', dict(returncode=run.returncode, at=time.time()))
            run.check_returncode()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(launch, range(4)))
    reports = [json.loads((OUT / f'f{f}.progress.json').read_text()) for f in DATA]
    assert all(r['status'] == 'complete' for r in reports)
    assert sum(r['n_questions'] for r in reports) == 1150
    put(OUT / 'result.json', dict(status='complete', completed=reports,
        checkpoint_sha256=WEIGHT, source_commit=COMMIT, finished_at=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--facts', type=int, choices=list(DATA))
    args = parser.parse_args()
    try:
        worker(args.facts) if args.facts else coordinate()
    except Exception as error:
        if OUT.exists():
            put(OUT / ('failure.json' if args.facts is None else f'failure_f{args.facts}.json'),
                dict(error=repr(error), at=time.time()))
        raise
