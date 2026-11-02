"""Small, read-only metadata catalog; listing never opens a training container."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml

from mahjong.contracts import MahjongError


def read_json(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(root) or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("metadata outside model root or too large")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("expected an object")
    return value


def manifest_identity(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(root) or path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("invalid manifest path or size")
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("expected manifest object")
    # Parse and hash exactly the same bytes, as FileCheckpointStore does.
    return value, hashlib.sha256(raw).hexdigest()


class Catalog:
    def __init__(self, root, *, weights_root=None):
        self.root = Path(root).resolve()
        self.weights_root = None if weights_root is None else Path(weights_root).resolve()
        self._entries = {}
        self.refresh()

    def refresh(self):
        runs, entries, errors = [], {}, []
        children = sorted(self.root.iterdir()) if self.root.is_dir() else []
        nested = self.root / "rl"
        if nested.is_dir() and nested.resolve().is_relative_to(self.root):
            children += sorted(nested.iterdir())
        for directory in children:
            if (
                directory.name.startswith(".")
                or not directory.is_dir()
                or not directory.resolve().is_relative_to(self.root)
            ):
                continue
            if not (directory / "run.json").exists():
                continue
            try:
                run = read_json(directory / "run.json", self.root)
                if not isinstance(run.get("run_id"), str):
                    raise ValueError("run_id missing")
                row = dict(
                    run_id=run["run_id"],
                    experiment=run.get("experiment", run["run_id"]),
                    created_at=run.get("created_at"),
                    training_type="unknown",
                    attempts=[],
                    checkpoints=[],
                    errors=[],
                )
                incompatible = False
                config = directory / "resolved_config.yaml"
                if config.exists():
                    if (
                        not config.resolve().is_relative_to(self.root)
                        or config.stat().st_size > 4 * 1024 * 1024
                    ):
                        raise ValueError("invalid run config")
                    value = yaml.safe_load(config.read_text()) or {}
                    if not isinstance(value, dict):
                        raise ValueError("run config must be an object")
                    data = value.get("data", {})
                    row["training_type"] = (
                        "RL"
                        if "rl" in value
                        else ("NTP" if "model" in value and data else "unknown")
                    )
                    model = value.get("model", {})
                    if isinstance(model, dict) and model:
                        incompatible = model.get("name") != "qwen3" or model.get(
                            "input_schema"
                        ) not in ("majang-policy-ntp/v1", "majang-policy-ntp/v2")
                attempts = directory / "attempts"
                for attempt in sorted(attempts.iterdir()) if attempts.is_dir() else []:
                    if not attempt.is_dir() or attempt.name.startswith("."):
                        continue
                    try:
                        metadata = read_json(attempt / "manifest.json", self.root)
                        status = (
                            read_json(attempt / "status.json", self.root)
                            if (attempt / "status.json").exists()
                            else {}
                        )
                        summary = (
                            read_json(attempt / "summary.json", self.root)
                            if (attempt / "summary.json").exists()
                            else {}
                        )
                        row["attempts"].append(
                            dict(
                                attempt_id=metadata["attempt_id"],
                                started_at=metadata.get("started_at"),
                                source_checkpoint=metadata.get("source_checkpoint"),
                                progress=status.get("progress"),
                                outcome=summary.get("completion", {}).get(
                                    "outcome", "unknown"
                                ),
                            )
                        )
                    except (
                        OSError,
                        ValueError,
                        KeyError,
                        TypeError,
                        AttributeError,
                        yaml.YAMLError,
                    ) as exc:
                        row["errors"].append(f"{attempt.name}: {exc}")
                checkpoints = directory / "checkpoints"
                for checkpoint in (
                    sorted(checkpoints.iterdir()) if checkpoints.is_dir() else []
                ):
                    if not checkpoint.is_dir() or checkpoint.name.startswith("."):
                        continue
                    try:
                        manifest, sha = manifest_identity(
                            checkpoint / "manifest.json", self.root
                        )
                        if (
                            manifest["run_id"] != run["run_id"]
                            or manifest["checkpoint_id"] != checkpoint.name
                        ):
                            raise ValueError("checkpoint identity mismatch")
                        if (
                            type(manifest["global_step"]) is not int
                            or manifest["global_step"] < 0
                        ):
                            raise ValueError("invalid checkpoint step")
                        identity = hashlib.sha256(
                            (str(checkpoint.resolve()) + ":" + sha).encode()
                        ).hexdigest()
                        reason = (
                            "首版仅支持 NTP checkpoint"
                            if row["training_type"] == "RL" or incompatible
                            else None
                        )
                        if not (checkpoint / "state.pt").is_file() or not (
                            checkpoint / "state.pt"
                        ).resolve().is_relative_to(self.root):
                            reason = "权重文件已删除"
                        item = dict(
                            id=identity,
                            checkpoint_id=manifest["checkpoint_id"],
                            attempt_id=manifest["attempt_id"],
                            step=manifest["global_step"],
                            created_at=manifest.get("created_at"),
                            selectable=reason is None,
                            status=reason or "待加载验证",
                        )
                        if reason is None:
                            entries[identity] = dict(
                                path=str(checkpoint.resolve()),
                                manifest_hash=sha,
                                run_id=run["run_id"],
                                checkpoint_id=manifest["checkpoint_id"],
                                attempt_id=manifest["attempt_id"],
                                step=manifest["global_step"],
                                state_sha256=manifest["state_sha256"],
                            )
                        row["checkpoints"].append(item)
                    except (
                        OSError,
                        ValueError,
                        KeyError,
                        TypeError,
                        AttributeError,
                        yaml.YAMLError,
                    ) as exc:
                        row["errors"].append(f"{checkpoint.name}: {exc}")
                row["checkpoints"].sort(key=lambda c: c["step"], reverse=True)
                runs.append(row)
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                AttributeError,
                yaml.YAMLError,
            ) as exc:
                errors.append(f"{directory.name}: {exc}")
        self._custom_weights(runs, entries, errors)
        if not self.root.is_dir() and not runs:
            errors.insert(0, "模型目录不存在，可直接使用随机对手。")
        self._entries = entries
        self.data = dict(runs=runs, errors=errors)
        return self.snapshot()

    def snapshot(self):
        return deepcopy(self.data)

    def _weight_signature(self, path):
        path = Path(path)
        if not path.resolve().is_relative_to(self.weights_root) or not path.is_file():
            raise ValueError("weights outside custom directory or removed")
        files = [path]
        sidecar = path.with_suffix('.json')
        if sidecar.exists():
            if not sidecar.resolve().is_relative_to(self.weights_root):
                raise ValueError("model config outside custom directory")
            files.append(sidecar)
        values = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns,
                   p.stat().st_ctime_ns, p.stat().st_ino) for p in files]
        return hashlib.sha256(json.dumps([str(path.absolute()), values]).encode()).hexdigest()

    def _custom_weights(self, runs, entries, errors):
        if self.weights_root is None or not self.weights_root.is_dir():
            return
        # Listing reads filenames and stat metadata only, never torch containers.
        for path in sorted(self.weights_root.iterdir()):
            if path.name.startswith('.') or path.suffix.lower() not in ('.pt', '.pth'):
                continue
            try:
                signature = self._weight_signature(path)
                identity = 'weights:' + signature
                run_id = 'weights:' + path.name
                checkpoint = dict(id=identity, checkpoint_id=path.name, attempt_id='',
                                  step=0, created_at=None, selectable=True, status="待加载验证")
                entries[identity] = dict(kind='weights', path=str(path.absolute()),
                                         manifest_hash=signature, run_id=run_id,
                                         checkpoint_id=path.name, attempt_id='',
                                         step=0, state_sha256=None)
                runs.append(dict(run_id=run_id, experiment=path.stem, created_at=None,
                                 training_type='Weights', attempts=[],
                                 checkpoints=[checkpoint], errors=[]))
            except (OSError, ValueError) as exc:
                errors.append(f'{path.name}: {exc}')

    def resolve(self, entry_id):
        try:
            entry = deepcopy(self._entries[entry_id])
            if entry.get('kind') == 'weights':
                if self._weight_signature(entry['path']) != entry['manifest_hash']:
                    raise ValueError("权重已变化，请刷新重选")
                return entry
            manifest, sha = manifest_identity(
                Path(entry["path"]) / "manifest.json", self.root
            )
            state = Path(entry["path"]) / "state.pt"
            if (
                sha != entry["manifest_hash"]
                or not state.is_file()
                or not state.resolve().is_relative_to(self.root)
            ):
                raise ValueError("模型已变化或被清理，请刷新重选")
            return entry
        except (KeyError, TypeError, OSError, ValueError) as exc:
            raise MahjongError(
                "CHECKPOINT_INVALID", f"无法使用所选模型，请刷新重选：{exc}"
            ) from exc
