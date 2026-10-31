"""Atomic report publishing with the public EvaluationReport schema."""
import os
from pathlib import Path
import tempfile

from mahjong.contracts import EvaluationReport, MahjongError, canonical_json, encode_json_contract


def report_reference(path: Path) -> str:
    if path.parent.parent.name not in {'reports', 'evaluations'}:
        raise MahjongError('EVALUATION_FAILED', 'Report path must be under reports/<attempt> or evaluations/<id>', 'evaluating')
    return Path(*path.parts[-3:]).as_posix()


def write_report(report: EvaluationReport, path: Path) -> None:
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f'Report identity already exists: {path}')
        fd, name = tempfile.mkstemp(prefix='.' + path.name + '.tmp-', dir=path.parent)
        temporary = Path(name)
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(canonical_json(encode_json_contract(report)) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except (OSError, MahjongError) as exc:
        raise MahjongError('EVALUATION_FAILED', f'Report write failed: {exc}', 'evaluating',
                          {'report_path': str(path)}) from exc
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink(missing_ok=True)
