"""Collect per-frame realtime timings and write comparable run reports."""

from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class FrameMetrics:
    frame_index: int
    elapsed_seconds: float
    read_ms: float
    inference_ms: float
    render_ms: float
    display_ms: float
    total_latency_ms: float
    fps: float
    detections: int


class Benchmark:
    def __init__(self, duration_seconds: float) -> None:
        self.duration_seconds = float(duration_seconds)
        self.started_at = 0.0
        self.finished_at = 0.0
        self.frames: list[FrameMetrics] = []
        self.dropped_frames = 0

    def start(self) -> None:
        self.frames.clear()
        self.dropped_frames = 0
        self.started_at = time.perf_counter()
        self.finished_at = 0.0
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass

    def elapsed(self) -> float:
        return (
            time.perf_counter() - self.started_at if self.started_at else 0.0
        )

    def should_stop(self) -> bool:
        return self.elapsed() >= self.duration_seconds

    def record_drop(self, count: int = 1) -> None:
        self.dropped_frames += max(0, int(count))

    def record(
        self,
        *,
        read_ms: float,
        inference_ms: float,
        render_ms: float,
        display_ms: float,
        total_latency_ms: float,
        detections: int,
    ) -> FrameMetrics:
        fps = 1000.0 / total_latency_ms if total_latency_ms > 0 else 0.0
        frame = FrameMetrics(
            frame_index=len(self.frames) + 1,
            elapsed_seconds=self.elapsed(),
            read_ms=float(read_ms),
            inference_ms=float(inference_ms),
            render_ms=float(render_ms),
            display_ms=float(display_ms),
            total_latency_ms=float(total_latency_ms),
            fps=fps,
            detections=int(detections),
        )
        self.frames.append(frame)
        return frame

    def overlay_metrics(self) -> tuple[float, float]:
        if not self.frames:
            return 0.0, 0.0
        recent = self.frames[-30:]
        average_latency = statistics.fmean(
            frame.total_latency_ms for frame in recent
        )
        fps = 1000.0 / average_latency if average_latency > 0 else 0.0
        return fps, self.frames[-1].total_latency_ms

    @staticmethod
    def _stats(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"average": None, "p95": None, "minimum": None, "maximum": None}
        return {
            "average": float(statistics.fmean(values)),
            "p95": float(np.percentile(values, 95)),
            "minimum": float(min(values)),
            "maximum": float(max(values)),
        }

    @staticmethod
    def _vram() -> dict[str, float | None]:
        try:
            import torch

            if not torch.cuda.is_available():
                return {
                    "peak_allocated_mb": None,
                    "peak_reserved_mb": None,
                }
            divisor = 1024.0 * 1024.0
            return {
                "peak_allocated_mb": torch.cuda.max_memory_allocated() / divisor,
                "peak_reserved_mb": torch.cuda.max_memory_reserved() / divisor,
            }
        except ImportError:
            return {"peak_allocated_mb": None, "peak_reserved_mb": None}

    def summary(
        self,
        *,
        stop_reason: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        self.finished_at = self.finished_at or time.perf_counter()
        runtime = max(0.0, self.finished_at - self.started_at)
        latencies = [frame.total_latency_ms for frame in self.frames]
        fps_values = [frame.fps for frame in self.frames]
        return {
            "status": "PASS" if self.frames else "FAIL",
            "stop_reason": stop_reason,
            "runtime_seconds": runtime,
            "processed_frames": len(self.frames),
            "dropped_frames": self.dropped_frames,
            "average_fps": len(self.frames) / runtime if runtime > 0 else 0.0,
            "minimum_fps": min(fps_values) if fps_values else None,
            "latency_ms": self._stats(latencies),
            "timing_ms": {
                "camera_read": self._stats(
                    [frame.read_ms for frame in self.frames]
                ),
                "inference": self._stats(
                    [frame.inference_ms for frame in self.frames]
                ),
                "render": self._stats(
                    [frame.render_ms for frame in self.frames]
                ),
                "display": self._stats(
                    [frame.display_ms for frame in self.frames]
                ),
            },
            "vram": self._vram(),
            "metadata": metadata,
        }

    def write(
        self,
        run_dir: Path,
        *,
        stop_reason: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=False)
        metrics_path = run_dir / "metrics.csv"
        fields = list(asdict(self.frames[0]).keys()) if self.frames else [
            field.name for field in FrameMetrics.__dataclass_fields__.values()
        ]
        with metrics_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for frame in self.frames:
                writer.writerow(asdict(frame))
        summary = self.summary(stop_reason=stop_reason, metadata=metadata)
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return summary
