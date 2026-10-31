"""Freeze raw input identities, semantic dependencies, and task membership."""
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path

from mahjong.tokenizer import MahjongTokenizer
from mahjong.tokenizer.codec import BINDING
from .common import (STORAGE_VERSION, TEXT_STORAGE_VERSION, TEXT_STORAGE_VERSIONS,
                     is_text_storage, digest, file_identity, require)


@dataclass(frozen=True)
class Source:
    source_key: str
    platform: str
    shard: str
    revision: str
    relative_path: str
    raw_bytes: int
    raw_sha256: str
    root: str
    locator: dict | None = None

    @property
    def path(self):
        require(self.locator is None, 'ARCHIVE_SOURCE', 'use open_source for TAR input')
        return Path(self.root) / self.relative_path

    def identity(self):
        return {k: getattr(self, k) for k in ('platform', 'shard', 'revision',
                'relative_path', 'raw_bytes', 'raw_sha256')}


def codec_binding():
    codec = MahjongTokenizer()
    return {**BINDING, 'codec_spec_sha256': codec.codec_spec_sha256}


def validate_catalog(catalog, *, check_implementation=True):
    binding = catalog['binding']
    backend = binding.get('backend', 'python')
    require(binding.get('compression') in (None, 'zstd/1', 'zstd-blocks/1') and
            (binding.get('compression') is None or binding.get('task_state') == 'append-state/1'),
            'UNSUPPORTED_STORAGE', 'compression binding')
    require(backend in ('python', 'rust'), 'UNSUPPORTED_BACKEND', backend)
    require((binding['storage'] == STORAGE_VERSION or is_text_storage(binding['storage'])) and binding['codec'] == codec_binding(),
            'BINDING_MISMATCH', 'storage/codec')
    if check_implementation:
        from .source_catalog import implementation_binding
        require(binding['implementation'] == implementation_binding(backend),
                'BINDING_MISMATCH', 'cleaning/environment implementation changed')
    sources = [Source(**s) for s in catalog['sources']]
    if 'task_state' in binding:
        require(binding['task_state'] == 'append-state/1' and
                binding['locators_digest'] == digest([s.locator for s in sources]),
                'CATALOG_CORRUPT', 'task state/source locators')
    require(all(s.source_key == digest(s.identity()) for s in sources),
            'CATALOG_CORRUPT', 'source identity')
    require(binding['input_digest'] == digest([s.identity() for s in sources]),
            'CATALOG_CORRUPT', 'input digest')
    require(catalog['semantic_id'] == digest(binding) and
            catalog['build_id'] == digest([catalog['semantic_id'], catalog['repair_revision']]),
            'CATALOG_CORRUPT', 'build identity')
    members = [m for t in catalog['tasks'] for m in t['members']]
    require(len(members) == len(set(members)) == len(sources) and
            set(members) == {s.source_key for s in sources} and
            [t['task_id'] for t in catalog['tasks']] == list(range(len(catalog['tasks']))),
            'CATALOG_CORRUPT', 'task membership')
    if check_implementation:
        from .profiles import verify_profile_resources
        verify_profile_resources(binding)


def __getattr__(name):
    """Resolve optional source-processing entry points only when requested."""
    if name in ('InputRoot', 'implementation_binding', 'plan', 'plan_sources', '_plan_sources'):
        from . import source_catalog
        value = getattr(source_catalog, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
