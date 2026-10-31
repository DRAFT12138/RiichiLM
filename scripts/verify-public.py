#!/usr/bin/env python3
"""CPU regression using only public source and synthetic runtime inputs.

Run from an installed editable checkout. No dataset, pretrained checkpoint,
external API, or persistent training is used. --integration-only skips units.
"""
import argparse
from dataclasses import replace
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]


def verify_imports():
    import mahjong, riichienv
    from riichienv import _riichienv
    assert Path(mahjong.__file__).resolve().is_relative_to(ROOT / 'src')
    assert not hasattr(riichienv, 'MjSoulReplay') and not hasattr(_riichienv, 'MjSoulReplay')
    for name, module in tuple(sys.modules.items()):
        if name == 'mahjong' or name.startswith('mahjong.'):
            filename = getattr(module, '__file__', None)
            if filename: assert Path(filename).resolve().is_relative_to(ROOT / 'src'), (name, filename)
    print('Public mahjong:', mahjong.__file__, flush=True)
    print('Rules without source parsers:', _riichienv.__file__, flush=True)


def invoke(*arguments):
    # Child processes inherit only the exported business source and installed
    # dependencies, including an optional native-wheel overlay on PYTHONPATH.
    program = '''import runpy,sys
from pathlib import Path
root=Path(sys.argv.pop(1));sys.path[:0]=[str(root/'src'),str(root)]
args=sys.argv[1:]
if args[0]=='cli':
 from mahjong.app.cli import main
 assert main(args[1:])==0
else:
 sys.argv=args[1:];runpy.run_path(sys.argv[0],run_name='__main__')
for name,module in tuple(sys.modules.items()):
 if name=='mahjong' or name.startswith('mahjong.'):
  filename=getattr(module,'__file__',None)
  if filename:assert Path(filename).resolve().is_relative_to(root/'src'),(name,filename)
'''
    subprocess.run([sys.executable, '-c', program, str(ROOT), *map(str, arguments)],
                   cwd=ROOT, check=True, timeout=900)


def latest(output):
    from mahjong.app.runs import active_attempt
    from mahjong.training.checkpoint import FileCheckpointStore
    run = next(output.iterdir())
    store = FileCheckpointStore()
    reference = store.find_latest(run, active_attempt(run))
    return run, reference, store.load(reference, run_dir=run)


def integration(output):
    from tests.support.policy import manual_build
    from tests.unit.training.test_runtime import assert_equal
    from mahjong.data.mahjong.corpus_pack import pack_corpus
    builds = [manual_build(output / f'input-{i}', f'synthetic-{i}') for i in range(2)]
    corpus = output / 'corpus'
    pack_corpus(builds, corpus, workers=1)
    def settings(directory, updates):
        return ['pretrain', f'output.root={directory}', f'training.max_updates={updates}',
                f'data.build_dir={corpus}', 'data.validation_fraction=0.5', 'data.validation_max_samples=2',
                'evaluation.nanikiru_dir=null', 'runtime.device=cpu', 'runtime.precision=32-true',
                'model.num_hidden_layers=1', 'model.hidden_size=8', 'model.intermediate_size=16',
                'model.num_attention_heads=1', 'model.num_key_value_heads=1', 'model.head_dim=8',
                'observability.tensorboard_enabled=false', 'checkpoint.every_seconds=null']
    invoke('cli', *settings(output / 'continuous', 2))
    invoke('cli', *settings(output / 'split', 1))
    run, ref, saved = latest(output / 'split')
    state = run / ref.path / 'state.pt'
    before = hashlib.sha256(state.read_bytes()).hexdigest()
    invoke('cli', 'resume', '--run', run, '--checkpoint', 'latest', 'training.max_updates=2')
    _, _, actual = latest(output / 'split')
    _, _, expected = latest(output / 'continuous')
    assert_equal(actual.runtime, expected.runtime); assert_equal(actual.rng, expected.rng); assert_equal(actual.data, expected.data)
    assert_equal(replace(actual.progress, last_evaluation_ref=None), replace(expected.progress, last_evaluation_ref=None))
    assert hashlib.sha256(state.read_bytes()).hexdigest() == before
    print('Qwen3 supervised update and cold resume: exact', flush=True)
    invoke('script', ROOT / 'scripts/verify-public-rl-resume.py', 'prepare', '--output', output / 'prepared.json')
    invoke('script', ROOT / 'scripts/verify-public-rl-resume.py', 'resume', '--reference', output / 'prepared.json', '--output', output / 'resumed.json')
    invoke('script', ROOT / 'scripts/verify-rl-cpu.py', '--output', output / 'selfplay')
    print('Public integration passed: synthetic training, PPO cold resume, two complete self-play matches', flush=True)


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--integration-only', action='store_true')
    parser.add_argument('--unit-only', action='store_true')
    options = parser.parse_args()
    for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'): os.environ[key] = ''
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '1'
    verify_imports()
    if not options.integration_only:
        import pytest
        result = pytest.main(['tests/unit', '-q', '--import-mode=importlib', '--disable-warnings'])
        if result: raise SystemExit(result)
    if not options.unit_only:
        with tempfile.TemporaryDirectory(prefix='riichilm-public-') as directory:
            integration(Path(directory))
    verify_imports()


if __name__ == '__main__': main()
