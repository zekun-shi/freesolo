import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('mode', ['error', 'partial', 'ok'])
def test_data_generator_requires_complete_output(tmp_path, mode):
    prepare = tmp_path / 'RULER source' / 'scripts' / 'data' / 'prepare.py'
    prepare.parent.mkdir(parents=True)
    prepare.write_text('''import json, os, pathlib, sys
args = sys.argv[1:]
value = lambda flag: args[args.index(flag) + 1]
if os.environ['FAKE_MODE'] == 'error':
    sys.exit(9)
folder = pathlib.Path(value('--save_dir')) / value('--task')
folder.mkdir(parents=True, exist_ok=True)
count = 99 if os.environ['FAKE_MODE'] == 'partial' else 100
with (folder / 'validation.jsonl').open('w') as stream:
    for _ in range(count):
        stream.write(json.dumps({'input':'question', 'outputs':['answer']}) + '\\n')
''')
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    (bin_dir / 'python').symlink_to(sys.executable)
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'], FAKE_MODE=mode,
               RULER_REPO=str(prepare.parents[2]), RULER_DATA=str(tmp_path / 'output data'),
               TASKS='niah_single_1', LENGTHS='4096')
    command = ['bash', str(ROOT / 'scripts/gen_ruler_data.sh')]
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=False, timeout=20)
    assert result.returncode == (0 if mode == 'ok' else 1), result.stdout + result.stderr
    assert ('GEN_ALL_DONE' in result.stdout) == (mode == 'ok')
    if mode == 'ok':
        result = subprocess.run(command, env=env, capture_output=True, text=True, check=False, timeout=20)
        assert result.returncode == 0 and 'HAVE complete' in result.stdout


@pytest.mark.parametrize('mode', ['error', 'partial', 'ok'])
def test_commonsense_grid_reports_worker_failures(tmp_path, mode):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    interpreter = bin_dir / 'python'
    interpreter.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
if not sys.argv[1].endswith('/eval_downstream.py'):
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
args = sys.argv[2:]
value = lambda flag: args[args.index(flag) + 1]
assert os.environ['FS_FUSED_ATTN'] == '0'
if os.environ['FAKE_MODE'] == 'error' and value('--tasks') == 'piqa':
    sys.exit(9)
report = {'config': {'checkpoint':value('--checkpoint'), 'config':value('--config'),
                    'limit':0, 'num_fewshot':0, 'max_seq_len':2048, 'no_rope':'--no_rope' in args},
          'tasks': {value('--tasks'): {'score':0.5, 'n':0 if os.environ['FAKE_MODE'] == 'partial' else 10}}}
pathlib.Path(value('--output')).write_text(json.dumps(report))
with open(os.environ['CALLS'],'a') as stream:
    stream.write('run\\n')
''')
    interpreter.chmod(0o755)
    calls = tmp_path / 'calls'
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'], FAKE_MODE=mode,
               CKPT=str(tmp_path / 'checkpoint'), OUT=str(tmp_path / 'results'),
               FS_FUSED_ATTN='0', CALLS=str(calls), MODE='fs', CONFIG='340m')
    command = ['bash', str(ROOT / 'scripts/eval_commonsense.sh')]
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=False, timeout=20)
    assert result.returncode == (0 if mode == 'ok' else 1), result.stdout + result.stderr
    assert ('CS_fs_DONE' in result.stdout) == (mode == 'ok')
    if mode == 'ok':
        result = subprocess.run(command, env=env, capture_output=True, text=True, check=False, timeout=20)
        assert result.returncode == 0
        assert result.stdout.count('HAVE complete') == 8
        assert calls.read_text().splitlines() == ['run'] * 8
