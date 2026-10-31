#!/usr/bin/env python3
"""Read-only environment inventory and explicit, real G1 probes.

``collect_environment()`` only collects metadata. The CLI additionally creates
probe artifacts and executes updates in fresh processes; it is not final G4
acceptance. Install the documented ROCm or CUDA requirements before running.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


from mahjong.environment import collect_environment, _filesystem


ROOT = Path(__file__).resolve().parents[1]


def _cpu_copy(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def _assert_equal(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for first, second in zip(left, right):
            _assert_equal(first, second)
    else:
        assert left == right


def _device_stage(device: str, stage: str, checkpoint: Path) -> dict:
    import torch
    from lightning_fabric import Fabric
    from lightning_fabric.strategies import SingleDeviceStrategy

    torch.set_num_threads(1)
    torch.manual_seed(20261011)
    raw_model = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
    raw_optimizer = torch.optim.Adam(raw_model.parameters(), lr=0.003)
    fabric = Fabric(accelerator="cpu" if device == "cpu" else "cuda", devices=1,
                    strategy=SingleDeviceStrategy(device=torch.device(device)), precision="32-true")
    fabric.launch()
    model, optimizer = fabric.setup(raw_model, raw_optimizer)
    assert model.module is raw_model
    wrapped_state = model.state_dict()
    _assert_equal(raw_model.state_dict(), wrapped_state)
    if stage == "reload":
        snapshot = torch.load(checkpoint, map_location="cpu", weights_only=True)
        raw_model.load_state_dict(snapshot["model"], strict=True)
        raw_optimizer.load_state_dict(snapshot["optimizer"])
        _assert_equal(snapshot["model"], raw_model.state_dict())
        _assert_equal(snapshot["optimizer"], raw_optimizer.state_dict())
        torch.set_rng_state(snapshot["cpu_rng"])
        if device != "cpu":
            torch.cuda.set_rng_state_all(snapshot["device_rng"])
    before = _cpu_copy(raw_model.state_dict())
    batch = torch.tensor([[1.0, 2.0], [-2.0, 1.0]], device=fabric.device)
    target = torch.tensor([[2.75], [-1.25]], device=fabric.device)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(batch), target)
    fabric.backward(loss)
    optimizer.step()
    if device != "cpu":
        torch.cuda.synchronize()
    after = _cpu_copy(raw_model.state_dict())
    assert torch.isfinite(loss)
    assert any(not torch.equal(before[key], after[key]) for key in before)
    assert len(raw_optimizer.state) > 0
    _assert_equal(raw_model.state_dict(), model.state_dict())
    _assert_equal(raw_optimizer.state_dict(), optimizer.state_dict())
    state_devices = {str(value.device) for state in raw_optimizer.state.values() for value in state.values() if isinstance(value, torch.Tensor)}
    if stage == "save":
        snapshot = {"model": after, "optimizer": _cpu_copy(raw_optimizer.state_dict()),
                    "cpu_rng": torch.get_rng_state().clone(),
                    "device_rng": [value.cpu().clone() for value in torch.cuda.get_rng_state_all()] if device != "cpu" else []}
        with checkpoint.open("wb") as stream:
            torch.save(snapshot, stream)
            stream.flush()
            os.fsync(stream.fileno())
    return {"device": device, "stage": stage, "pid": os.getpid(), "loss": float(loss.detach().cpu()),
            "parameters_changed": True, "optimizer_initialized": True, "raw_wrapped_model_equal": True,
            "raw_wrapped_optimizer_equal": True, "optimizer_tensor_devices": sorted(state_devices),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "strategy": type(fabric.strategy).__name__, "precision": type(fabric._precision).__name__}


def _device_probe(device: str, path: Path) -> dict:
    checkpoint = path / ("cpu-state.pt" if device == "cpu" else "gpu-state.pt")
    results = []
    for stage in ("save", "reload"):
        command = [sys.executable, str(Path(__file__).resolve()), "--device-stage", stage, "--device", device, "--checkpoint", str(checkpoint)]
        proc = subprocess.run(command, text=True, capture_output=True, check=True)
        result = json.loads(proc.stdout)
        result["command"] = command
        result["stderr"] = proc.stderr
        results.append(result)
    assert results[0]["pid"] != results[1]["pid"]
    return {"status": "passed", "stages": results, "new_process_reload": True}


def _config_probe(path: Path) -> dict:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from omegaconf.errors import ConfigAttributeError, ConfigKeyError, ValidationError

    @dataclass
    class Schema:
        steps: int = 200

    config_dir = path / "config"
    config_dir.mkdir()
    (config_dir / "probe.yaml").write_text("optimizer:\n  lr: 0.003\ntraining:\n  max_updates: 200\n")
    original_cwd = Path.cwd()
    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base="1.3"):
        cfg = compose(config_name="probe", overrides=["optimizer.lr=0.006", "training.max_updates=23"])
    assert cfg.optimizer.lr == 0.006 and cfg.training.max_updates == 23
    assert Path.cwd() == original_cwd
    typed = OmegaConf.structured(Schema)
    rejected = []
    for key, value in [("unknown", 1), ("steps", "not-an-integer")]:
        try:
            typed[key] = value
        except (ConfigAttributeError, ConfigKeyError, ValidationError):
            rejected.append(key)
        else:
            raise AssertionError(f"Invalid structured config accepted: {key}")
    return {"status": "passed", "composed": OmegaConf.to_container(cfg), "cwd_unchanged": True, "rejected": rejected}


def _tensorboard_probe(path: Path) -> dict:
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    log_dir = path / "tensorboard"
    writer = SummaryWriter(str(log_dir), max_queue=256, flush_secs=2)
    writer.add_scalar("environment/loss", 2.5, 1)
    writer.flush()
    # Read while the writer is still open: this probes visibility, not just close.
    reader = EventAccumulator(str(log_dir)).Reload()
    event = reader.Scalars("environment/loss")[0]
    assert event.step == 1 and event.value == 2.5
    writer.close()
    return {"status": "passed", "value": event.value, "step": event.step, "read_before_close": True, "log_dir": str(log_dir)}


def _filesystem_probe(path: Path) -> dict:
    directory = path / "filesystem"
    directory.mkdir()
    final = directory / "state"
    old = b"previous-complete-state"
    new = b"new-complete-state"
    with final.open("wb") as stream:
        stream.write(old)
        stream.flush()
        os.fsync(stream.fileno())
    temporary = directory / ".state.tmp"
    with temporary.open("wb") as stream:
        stream.write(new)
        stream.flush()
        os.fsync(stream.fileno())
    # A failure before publication leaves the complete prior file unchanged.
    try:
        raise OSError("controlled pre-publication failure")
    except OSError:
        assert final.read_bytes() == old
    os.replace(temporary, final)
    descriptor = os.open(directory, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    assert final.read_bytes() == new and not temporary.exists()
    return {"status": "passed", "file_fsync": True, "directory_fsync": True,
            "atomic_replace": True, "pre_publication_old_state_protected": True,
            "filesystem": _filesystem(directory), "published_sha256": hashlib.sha256(new).hexdigest()}


def _pytest_probe(path: Path) -> dict:
    test = path / "test_environment.py"
    test.write_text("def test_real_cpu_tensor():\n    import torch\n    assert torch.tensor([1, 2]).sum().item() == 3\n")
    command = [sys.executable, "-m", "pytest", str(test), "-q", "-p", "no:cacheprovider"]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    return {"status": "passed", "command": command, "stdout": result.stdout, "stderr": result.stderr}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "environment-probe")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--gpu", action="store_true", help="run exclusive target GPU update/reload probe")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--device-stage", choices=("save", "reload"), help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cpu", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.device_stage:
        print(json.dumps(_device_stage(args.device, args.device_stage, args.checkpoint)))
        return 0
    report = collect_environment(args.output_root)
    if report["protected_mismatches"]:
        raise RuntimeError(f"Protected GPU packages differ: {report['protected_mismatches']}")
    if not args.inventory_only:
        args.output_root.mkdir(parents=True, exist_ok=True)
        path = Path(tempfile.mkdtemp(prefix="probe-", dir=args.output_root)).resolve()
        report["probe_artifact_root"] = str(path)
        report["probes"] = {
            "cpu": _device_probe("cpu", path),
            "configuration": _config_probe(path),
            "tensorboard": _tensorboard_probe(path),
            "filesystem": _filesystem_probe(path),
            "pytest": _pytest_probe(path),
        }
        if args.gpu and report["torch"]["cuda_api_available"]:
            report["probes"]["gpu"] = _device_probe("cuda:0", path)
        else:
            report["probes"]["gpu"] = {"status": "unverified", "reason": "not_requested" if not args.gpu else "device_unavailable", "followup_owner": "P10"}
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized)
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
