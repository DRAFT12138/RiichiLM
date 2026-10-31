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
    def __init__(self, root):
        self.root = Path(root).resolve()
        self._entries = {}
        self.refresh()

    def refresh(self):
        runs, entries, errors = [], {}, []
        if not self.root.is_dir():
            self._entries = {}
            self.data = dict(runs=[], errors=["模型目录不存在，可直接使用随机对手。"])
            return self.snapshot()
        children = sorted(self.root.iterdir())
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
        self._entries = entries
        self.data = dict(runs=runs, errors=errors)
        return self.snapshot()

    def snapshot(self):
        return deepcopy(self.data)

    def resolve(self, entry_id):
        try:
            entry = deepcopy(self._entries[entry_id])
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
