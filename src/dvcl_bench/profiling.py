"""Auditable model-efficiency profiling shared by all training backends."""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from itertools import chain
from typing import Callable, Optional

import torch

from .specs import ProfilingSpec


_CURRENT_SESSION: ContextVar[Optional["ProfilingSession"]] = ContextVar(
    "dvcl_profiling_session", default=None
)


@dataclass
class ProfilingSession:
    spec: ProfilingSpec
    device: torch.device
    started_at: float = 0.0
    elapsed_seconds: float = 0.0
    inference_overhead_seconds: float = 0.0
    inference_latencies_ms: list[float] = field(default_factory=list)
    trainable_parameters: Optional[int] = None
    total_parameters: Optional[int] = None
    parameter_bytes: Optional[int] = None
    peak_allocated_bytes: Optional[int] = None
    peak_reserved_bytes: Optional[int] = None

    def start(self) -> None:
        _synchronize(self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.started_at = time.perf_counter()

    def stop(self) -> None:
        _synchronize(self.device)
        self.elapsed_seconds = time.perf_counter() - self.started_at
        if self.device.type == "cuda" and self.peak_allocated_bytes is None:
            self.peak_allocated_bytes = int(
                torch.cuda.max_memory_allocated(self.device)
            )
            self.peak_reserved_bytes = int(
                torch.cuda.max_memory_reserved(self.device)
            )

    def measure_inference(self, model, forward: Callable[[], object]) -> None:
        if self.trainable_parameters is not None:
            raise RuntimeError("profiling reported more than one model")
        started = time.perf_counter()
        models = tuple(model) if isinstance(model, (tuple, list)) else (model,)
        parameters = list(chain.from_iterable(
            current.parameters() for current in models
        ))
        self.trainable_parameters = sum(
            parameter.numel() for parameter in parameters
            if parameter.requires_grad
        )
        self.total_parameters = sum(parameter.numel() for parameter in parameters)
        self.parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in parameters
        )
        if self.device.type == "cuda":
            _synchronize(self.device)
            self.peak_allocated_bytes = int(
                torch.cuda.max_memory_allocated(self.device)
            )
            self.peak_reserved_bytes = int(
                torch.cuda.max_memory_reserved(self.device)
            )
        for current in models:
            current.eval()
        with torch.no_grad():
            for _ in range(self.spec.inference_warmup):
                output = forward()
                del output
            _synchronize(self.device)
            for _ in range(self.spec.inference_repetitions):
                _synchronize(self.device)
                iteration_started = time.perf_counter()
                output = forward()
                _synchronize(self.device)
                self.inference_latencies_ms.append(
                    1000.0 * (time.perf_counter() - iteration_started)
                )
                del output
        self.inference_overhead_seconds += time.perf_counter() - started

    def summary(self, history) -> dict:
        if self.trainable_parameters is None or not self.inference_latencies_ms:
            raise RuntimeError("trainer did not report an inference model")
        training_seconds = self.elapsed_seconds - self.inference_overhead_seconds
        if training_seconds <= 0:
            raise RuntimeError("profiled training time must be positive")
        iterations = len(history)
        if iterations <= 0:
            raise RuntimeError("profiled training history must not be empty")
        return {
            "scope": "trainer_pipeline_excluding_profile_repetitions",
            "device": str(self.device),
            "trainable_parameters": self.trainable_parameters,
            "total_parameters": self.total_parameters,
            "parameter_bytes": self.parameter_bytes,
            "training_seconds": training_seconds,
            "training_iterations": iterations,
            "seconds_per_iteration": training_seconds / iterations,
            "inference_warmup": self.spec.inference_warmup,
            "inference_repetitions": self.spec.inference_repetitions,
            "inference_latency_ms_mean": statistics.fmean(
                self.inference_latencies_ms
            ),
            "inference_latency_ms_std": (
                statistics.stdev(self.inference_latencies_ms)
                if len(self.inference_latencies_ms) > 1 else 0.0
            ),
            "inference_latency_ms_median": statistics.median(
                self.inference_latencies_ms
            ),
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
        }


@contextmanager
def profile_run(spec: ProfilingSpec, device: str):
    if not spec.enabled:
        yield None
        return
    session = ProfilingSession(spec=spec, device=torch.device(device))
    token = _CURRENT_SESSION.set(session)
    session.start()
    try:
        yield session
    finally:
        session.stop()
        _CURRENT_SESSION.reset(token)


def profile_inference(model, forward: Callable[[], object]) -> None:
    session = _CURRENT_SESSION.get()
    if session is not None:
        session.measure_inference(model, forward)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
