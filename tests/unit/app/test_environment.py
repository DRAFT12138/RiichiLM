"""Inventory follows the installed GPU build without requiring a visible GPU."""
import pytest
import torch
from packaging.requirements import Requirement
import tomllib

from mahjong import environment


@pytest.mark.parametrize('hip,variant', [('10.1.0', 'rocm'), (None, 'cuda')])
def test_inventory_selects_build_and_uses_its_pins(monkeypatch, hip, variant):
    monkeypatch.setattr(torch.version, 'hip', hip)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    report = environment.collect_environment()
    filename = f'requirements-{variant}.txt'
    assert report['requirements_file'] == filename
    assert report['torch']['devices'] == []
    assert set(report['source_file_sha256']) >= {'requirements-rocm.txt', 'requirements-cuda.txt'}
    sources = environment._locked_sources(filename)
    assert sources['torch']['version'] == ('2.14.0+rocm10.1.0' if hip else '2.14.0+cu130')
    for name, mismatch in report['protected_mismatches'].items():
        assert mismatch['expected'] == sources[name]['version']
    if not hip:
        assert not any(name.startswith(('rocm', 'amd-')) for name in sources)


def test_project_metadata_accepts_both_torch_builds():
    project = tomllib.loads((environment.ROOT / 'pyproject.toml').read_text())['project']
    requirement = next(Requirement(value) for value in project['dependencies']
                       if Requirement(value).name == 'torch')
    for version in ('2.14.0+rocm10.1.0', '2.14.0+cu130'):
        assert version in requirement.specifier
