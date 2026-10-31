"""Optional user-supplied benchmarks preserve legacy saved configuration trees."""
from copy import deepcopy
from pathlib import Path
import pytest
from mahjong.app import config, assembly
from mahjong.contracts import MahjongError


def test_old_checkpoint_tree_and_integrity_basis_are_not_migrated(tmp_path):
    saved = config.parse_new([], tmp_path)
    del saved['evaluation']['nanikiru_dir']
    before = deepcopy(saved)
    resumed, changes = config.merge_resume(saved, ['training.max_updates=201'])
    assert resumed['evaluation'] == before['evaluation']
    assert config.frozen_config(resumed) == config.frozen_config(before)
    assert saved == before and changes


@pytest.mark.parametrize('value', ['null', '/tmp/user-evaluation'])
def test_explicit_user_input_is_accepted_and_frozen(tmp_path, value):
    resolved = config.parse_new([f'evaluation.nanikiru_dir={value}'], tmp_path)
    assert resolved['evaluation']['nanikiru_dir'] == (None if value == 'null' else value)
    with pytest.raises(MahjongError):
        config.merge_resume(resolved, ['evaluation.nanikiru_dir=null'])


@pytest.mark.parametrize('value', ['true', '42', '[]', '""'])
def test_invalid_optional_benchmark_values_fail(tmp_path, value):
    with pytest.raises(MahjongError, match='evaluation.nanikiru_dir'):
        config.parse_new([f'evaluation.nanikiru_dir={value}'], tmp_path)
