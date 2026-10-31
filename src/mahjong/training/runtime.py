"""The only Fabric adapter; models and data retain their own semantics."""

from contextlib import contextmanager
from datetime import datetime, timezone
import random

import numpy as np
import torch
from lightning_fabric import Fabric
from lightning_fabric.strategies import SingleDeviceStrategy

from mahjong import contracts as c
from mahjong.training.state import decode_numpy_rng, encode_numpy_rng, seeded_rng


class _ForwardBridge(torch.nn.Module):
    """Preserve public immutable results across Fabric's mutable traversal.

    Fabric recursively registers hooks on its forward output and cannot walk
    frozen dataclasses. This adapter is private to the runtime; the model and
    all consumers retain the documented StepResult contract.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, payload, sample_ids, sample_count, input_schema, *, phase):
        # Fabric AMP converts inputs recursively and cannot mutate frozen Batch
        # values. Cross the Fabric boundary as plain values, then reconstruct.
        batch = c.Batch(payload, sample_ids, sample_count, input_schema)
        result = self.model(batch, phase=phase)
        c.validate_step_result(result, phase=phase)
        return result.loss, {name: (stat.numerator, stat.denominator, stat.unit)
                             for name, stat in result.metrics.items()}, result.skip_reason


class FabricRuntime:
    def __init__(self, device="cpu", precision="32-true", deterministic=True, cpu_threads=1):
        if device not in {"cpu", "cuda:0"} or precision not in {"32-true", "bf16-mixed"}:
            raise c.MahjongError("CONFIG_INVALID", "Runtime supports cpu/cuda:0 and 32-true/bf16-mixed")
        if type(deterministic) is not bool or type(cpu_threads) is not int or cpu_threads < 1:
            raise c.MahjongError("CONFIG_INVALID", "Invalid deterministic or cpu_threads setting")
        if device == "cuda:0" and not torch.cuda.is_available():
            raise c.MahjongError("DEVICE_UNAVAILABLE", "Requested cuda:0 is unavailable")
        if device == "cuda:0" and precision == "bf16-mixed" and not torch.cuda.is_bf16_supported():
            raise c.MahjongError("DEVICE_UNAVAILABLE", "Requested GPU does not support BF16")
        self.device = torch.device(device)
        self.precision = precision
        torch.set_num_threads(cpu_threads)
        torch.use_deterministic_algorithms(deterministic)
        self.fabric = Fabric(accelerator="cpu" if device == "cpu" else "cuda", devices=1,
                             strategy=SingleDeviceStrategy(device=self.device), precision=precision)
        self.raw_model = self.raw_optimizer = self.model = self.optimizer = None
        # Scratch is transient, reset on every check, and never checkpointed.
        self._gradient_check_scratch = {}
        self._cpu_batch_preview = None

    def setup(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None) -> None:
        if self.raw_model is not None:
            raise c.MahjongError("CONTRACT_MISMATCH", "Runtime.setup must be called once")
        execution = f"{self.device}/{self.precision}"
        if self.precision == "bf16-mixed" and execution not in model.descriptor().supported_execution:
            raise c.MahjongError("CONFIG_INVALID", f"Model does not support {execution}")
        self.raw_model, self.raw_optimizer = model, optimizer
        self.fabric.launch()
        bridge = _ForwardBridge(model)
        if optimizer is None:
            self.model = self.fabric.setup_module(bridge)
        else:
            self.model, self.optimizer = self.fabric.setup(bridge, optimizer)

    def descriptor(self) -> c.ComponentDescriptor:
        return self.raw_model.descriptor()

    def configure_training_execution(self, remaining_updates: int) -> None:
        self.discard_batch_preview()
        configure = getattr(self.raw_model, 'configure_training_execution', None)
        if configure is not None:
            configure(remaining_updates)

    def compute(self, batch: c.Batch, *, phase: str) -> c.StepResult:
        preview, self._cpu_batch_preview = self._cpu_batch_preview, None
        if (phase == 'train' and preview is not None and preview[0] is batch and
                self.supports_cpu_lookahead() and preview[1] == _batch_version(batch)):
            if isinstance(preview[2], Exception):
                raise preview[2]
            prepared = self.raw_model.transfer_prepared_batch(preview[2], self.device)
        else:
            prepared = self.raw_model.prepare_batch(batch, self.device)
        loss, metrics, skip_reason = self.model(prepared.payload, prepared.sample_ids,
            prepared.sample_count, prepared.input_schema, phase=phase)
        result = c.StepResult(loss, {name: c.MetricStat(*stat) for name, stat in metrics.items()}, skip_reason)
        c.validate_step_result(result, phase=phase)
        return result

    def supports_cpu_lookahead(self):
        supported = getattr(self.raw_model, 'supports_cpu_lookahead', None)
        return (self.device.type == 'cuda' and self.precision == 'bf16-mixed' and
                self.raw_optimizer is not None and supported is not None and supported())

    def discard_batch_preview(self):
        self._cpu_batch_preview = None

    def prepare_cpu_batch(self, batch):
        self.discard_batch_preview()
        if not self.supports_cpu_lookahead():
            return
        version = _batch_version(batch)
        try:
            prepared = self.raw_model.prepare_batch(batch, torch.device('cpu'))
        except Exception as error:
            prepared = error.with_traceback(None)
        self._cpu_batch_preview = (batch, version, prepared)

    def zero_grad(self) -> None:
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)

    def backward(self, loss: torch.Tensor) -> None:
        self.fabric.backward(loss)

    @torch.no_grad()
    def check_gradients(self) -> None:
        gradients = [parameter.grad for parameter in self.raw_model.parameters()
                     if parameter.requires_grad and parameter.grad is not None]
        if not gradients:
            raise c.MahjongError("CONTRACT_MISMATCH", "Training result produced no parameter gradients", "training")
        # The locked PyTorch build exposes the AMP multi-tensor finite scan.
        # A unit inverse scale preserves FP32/FP16 gradients and needs no scaler.
        # BF16 mixed training has FP32 gradients; this ROCm primitive does not
        # support native BF16 gradients, which use the conventional fallback.
        # Fall back for other layouts/dtypes or builds without this primitive.
        scan = getattr(torch, '_amp_foreach_non_finite_check_and_unscale_', None)
        buckets, fallback = {}, []
        for gradient in gradients:
            if (scan is not None and gradient.device.type == 'cuda' and
                    gradient.layout == torch.strided and gradient.is_contiguous() and
                    gradient.dtype in (torch.float16, torch.float32)):
                buckets.setdefault((gradient.device, gradient.dtype), []).append(gradient)
            else:
                fallback.append(gradient)
        checked = set()
        for (device, _), values in buckets.items():
            if device not in self._gradient_check_scratch:
                self._gradient_check_scratch[device] = (
                    torch.zeros((), device=device, dtype=torch.float32),
                    torch.ones((), device=device, dtype=torch.float32))
            found, inverse_scale = self._gradient_check_scratch[device]
            if device not in checked:
                found.zero_()
                checked.add(device)
            scan(values, found, inverse_scale)
        nonfinite = any(bool(self._gradient_check_scratch[device][0]) for device in checked)
        if nonfinite or any(not bool(torch.isfinite(
                gradient._values() if gradient.layout == torch.sparse_coo else gradient).all())
                for gradient in fallback):
            raise c.MahjongError("NONFINITE_GRADIENT", "Training produced a nonfinite gradient", "training")

    def step(self) -> None:
        if self.optimizer is None:
            raise c.MahjongError("CONTRACT_MISMATCH", "Evaluation runtime has no optimizer", "training")
        self.optimizer.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def learning_rates(self) -> tuple[float, ...]:
        return () if self.raw_optimizer is None else tuple(float(group["lr"]) for group in self.raw_optimizer.param_groups)

    def export_state(self) -> c.RuntimeState:
        return c.RuntimeState(c.clone_state(self.raw_model.state_dict()), self.raw_model.auxiliary_state_dict(),
                              None if self.raw_optimizer is None else c.clone_state(self.raw_optimizer.state_dict()), None, None)

    def import_state(self, state: c.RuntimeState, *, model_only=False) -> None:
        self.discard_batch_preview()
        c.validate_contract(state)
        # BF16 autocast retains FP32 master parameters and Adam state and does
        # not use loss scaling. Never silently ignore incompatible scaler state.
        if not model_only and (state.scaler is not None or state.scheduler is not None):
            raise c.MahjongError("RESUME_INCOMPATIBLE", "Unexpected scaler/scheduler state", "restoring")
        try:
            self.raw_model.load_state_dict(state.model, strict=True)
            self.raw_model.load_auxiliary_state_dict(state.auxiliary)
            if not model_only:
                if self.raw_optimizer is None or state.optimizer is None:
                    raise c.MahjongError("RESUME_INCOMPATIBLE", "Full recovery requires an optimizer state", "restoring")
                optimizer_state = c.clone_state(state.optimizer)
                if (self.device.type == "cuda" and isinstance(self.raw_optimizer, torch.optim.Adam)
                        and self.raw_optimizer.defaults.get("fused") is True):
                    # Resume GPU NTP with its current execution policy, retaining
                    # every saved hyperparameter, moment and step value. Set the
                    # flags before loading so PyTorch also places step counters
                    # on the parameter device, as required by fused Adam.
                    for group in optimizer_state["param_groups"]:
                        group.update(fused=True, foreach=False)
                self.raw_optimizer.load_state_dict(optimizer_state)
        except (ValueError, RuntimeError) as exc:
            if isinstance(exc, c.MahjongError):
                raise
            raise c.MahjongError("RESUME_INCOMPATIBLE", str(exc), "restoring") from exc

    def reset_auxiliary_rng(self, seed: int) -> None:
        self.raw_model.reset_auxiliary_rng(seed)

    def capture_rng(self) -> c.RngState:
        return c.RngState(c.clone_state(random.getstate()), encode_numpy_rng(np.random.get_state()),
                          torch.get_rng_state().clone(),
                          torch.cuda.get_rng_state(self.device).cpu().clone() if self.device.type == "cuda" else None, {})

    def restore_rng(self, state: c.RngState) -> None:
        c.validate_contract(state)
        if state.runtime_private:
            raise c.MahjongError("RESUME_INCOMPATIBLE", "Unexpected runtime private RNG state", "restoring")
        if (state.torch_device is None) != (self.device.type == "cpu"):
            raise c.MahjongError("RESUME_INCOMPATIBLE", "RNG device class differs", "restoring")
        random.setstate(state.python)
        if state.numpy is not None:
            np.random.set_state(decode_numpy_rng(state.numpy))
        torch.set_rng_state(state.torch_cpu)
        if state.torch_device is not None:
            torch.cuda.set_rng_state(state.torch_device, self.device)

    def seed_training(self, seed: int) -> None:
        self.restore_rng(seeded_rng(seed, self.capture_rng()))

    @contextmanager
    def evaluation_scope(self, seed: int):
        self.discard_batch_preview()
        rng = self.capture_rng()
        auxiliary = self.raw_model.auxiliary_state_dict()
        modes = [(module, module.training) for module in self.raw_model.modules()]
        gradients = [(parameter, None if parameter.grad is None else parameter.grad.detach().clone())
                     for parameter in self.raw_model.parameters()]
        try:
            self.seed_training(seed)
            self.reset_auxiliary_rng(seed)
            self.raw_model.eval()
            with torch.inference_mode():
                yield
        finally:
            self.discard_batch_preview()
            self.raw_model.load_auxiliary_state_dict(auxiliary)
            for module, training in modes:
                module.training = training
            for parameter, gradient in gradients:
                parameter.grad = gradient
            self.restore_rng(rng)

    def resource_snapshot(self) -> c.ResourceSnapshot:
        utc = datetime.now(timezone.utc).isoformat()
        values = {}
        for name, getter in (("allocated", torch.cuda.memory_allocated), ("reserved", torch.cuda.memory_reserved)):
            if self.device.type == "cpu":
                values[name] = c.ResourceValue(None, "bytes", "unavailable", "cpu_execution", utc, 0.)
            else:
                try:
                    values[name] = c.ResourceValue(int(getter(self.device)), "bytes", "available", None, utc, 0.)
                except (RuntimeError, OSError) as exc:
                    values[name] = c.ResourceValue(None, "bytes", "unavailable", str(exc), utc, 0.)
        return c.ResourceSnapshot(str(self.device), utc, values)


def _batch_version(batch):
    """Input mutation guard for the privately owned, CPU-only raw batch."""
    return (batch.sample_ids, batch.sample_count, batch.input_schema,
            tuple((key, (id(value), value._version, value.shape, value.dtype, value.device)
                   if isinstance(value, torch.Tensor) else value)
                  for key, value in batch.payload.items()))
