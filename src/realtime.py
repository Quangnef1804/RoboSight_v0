"""Realtime RF-DETR camera inference and performance benchmarking."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .realtime_pipeline.benchmark import Benchmark
from .realtime_pipeline.camera import Camera
from .realtime_pipeline.detector import Detector
from .realtime_pipeline.renderer import Renderer


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "realtime.yaml"


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def load_config(
    path: Path, *, require_model_checkpoint: bool = True
) -> dict[str, Any]:
    config_path = path.resolve()
    config = _load_mapping(config_path)
    required = {
        "dataset_config",
        "model",
        "camera",
        "inference",
        "benchmark",
        "display",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Missing realtime config sections: {missing}")

    dataset_path = _resolve(config["dataset_config"])
    dataset = _load_mapping(dataset_path)
    classes = dataset.get("classes")
    if (
        not isinstance(classes, list)
        or not classes
        or len(classes) != len(set(classes))
    ):
        raise ValueError("dataset_config classes must be a non-empty unique list.")
    dataset["_config_path"] = str(dataset_path)
    config["dataset_config"] = str(dataset_path)
    config["dataset"] = dataset

    for section in ("model", "camera", "inference", "benchmark", "display"):
        if not isinstance(config[section], dict):
            raise ValueError(f"{section} must be a mapping.")

    model = config["model"]
    for field in ("name", "checkpoint", "device", "expected_class_count"):
        if field not in model:
            raise ValueError(f"Missing model.{field}")
    checkpoint = _resolve(model["checkpoint"])
    if checkpoint.suffix.lower() != ".pth":
        raise ValueError("model.checkpoint must point to a .pth checkpoint.")
    if require_model_checkpoint and not checkpoint.is_file():
        raise FileNotFoundError(f"RF-DETR checkpoint not found: {checkpoint}")
    model["checkpoint"] = str(checkpoint)
    expected_count = int(model["expected_class_count"])
    if len(classes) != expected_count:
        raise ValueError(
            f"dataset classes count is {len(classes)}, expected {expected_count}."
        )
    device = str(model["device"]).lower()
    if not (device == "cpu" or device.startswith("cuda")):
        raise ValueError("model.device must be cpu or cuda.")

    camera = config["camera"]
    for field in ("id", "resolution"):
        if field not in camera:
            raise ValueError(f"Missing camera.{field}")
    resolution = camera["resolution"]
    if not isinstance(resolution, dict):
        raise ValueError("camera.resolution must be a mapping.")
    width = int(resolution.get("width", 0))
    height = int(resolution.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("camera resolution must be positive.")
    resolution.update(width=width, height=height)
    codec = str(camera.get("codec", "MJPG")).upper()
    if len(codec) != 4:
        raise ValueError("camera.codec must contain exactly four characters.")
    camera["codec"] = codec

    threshold = float(
        config["inference"].get("confidence_threshold", -1)
    )
    if not 0 <= threshold <= 1:
        raise ValueError("inference.confidence_threshold must be between 0 and 1.")
    config["inference"]["confidence_threshold"] = threshold
    roi = config["inference"].get("roi", [0.0, 0.0, 1.0, 1.0])
    if not isinstance(roi, list) or len(roi) != 4:
        raise ValueError("inference.roi must contain [x, y, width, height].")
    roi = [float(value) for value in roi]
    roi_x, roi_y, roi_width, roi_height = roi
    if (
        roi_width <= 0
        or roi_height <= 0
        or roi_x < 0
        or roi_y < 0
        or roi_x + roi_width > 1
        or roi_y + roi_height > 1
    ):
        raise ValueError("inference.roi must be a normalized rectangle inside the frame.")
    config["inference"]["roi"] = roi

    benchmark = config["benchmark"]
    warmup_frames = int(benchmark.get("warmup_frames", 0))
    if not 10 <= warmup_frames <= 30:
        raise ValueError("benchmark.warmup_frames must be between 10 and 30.")
    duration = float(benchmark.get("duration_seconds", 0))
    if duration <= 0:
        raise ValueError("benchmark.duration_seconds must be positive.")
    benchmark["warmup_frames"] = warmup_frames
    benchmark["duration_seconds"] = duration
    benchmark["output_root"] = str(_resolve(benchmark["output_root"]))

    config["_config_path"] = str(config_path)
    return config


def _run_dir(config: dict[str, Any], phase: str = "realtime") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{phase}_{stamp}_{uuid.uuid4().hex[:6]}"
    return Path(config["benchmark"]["output_root"]) / run_id


def _capture_fps(
    start: dict[str, float | int],
    end: dict[str, float | int],
) -> float:
    elapsed = float(end["timestamp"]) - float(start["timestamp"])
    captured = int(end["captured_frames"]) - int(start["captured_frames"])
    return captured / elapsed if elapsed > 0 else 0.0


def _capture_diagnostics(
    properties: dict[str, Any],
    measured_fps: float,
    target_fps: float,
) -> list[str]:
    diagnostics: list[str] = []
    requested_codec = str(properties.get("requested_codec", "")).upper()
    actual_codec = str(properties.get("actual_codec", "")).upper()
    if requested_codec and actual_codec and requested_codec != actual_codec:
        diagnostics.append(
            f"Driver returned codec {actual_codec} instead of {requested_codec}."
        )
    if measured_fps < target_fps * 0.8:
        diagnostics.append(
            "Capture is below 80% of target; check lighting/auto exposure, "
            "manual exposure, USB bandwidth, or a lower resolution."
        )
    if float(properties.get("auto_exposure", -1.0)) < 0:
        diagnostics.append(
            "The selected backend does not expose auto-exposure state."
        )
    return diagnostics


def _warmup(
    camera: Camera,
    detector: Detector | None,
    frames: int,
) -> dict[str, int]:
    completed = 0
    skipped = 0
    failed = 0
    deadline = time.perf_counter() + max(10.0, float(frames))
    label = "RF-DETR" if detector is not None else "camera"
    print(f"Warming up {label} for {frames} frames...")
    while completed < frames:
        result = camera.snapshot()
        skipped += result.skipped_frames
        failed += result.failed_reads
        if result.frame is None:
            if time.perf_counter() >= deadline:
                raise TimeoutError("Camera produced no usable frames during warm-up.")
            continue
        if detector is not None:
            detector.predict(result.frame)
        completed += 1
        print(f"\rWarm-up {completed}/{frames}", end="", flush=True)
    print()
    return {"skipped_frames": skipped, "failed_reads": failed}


def run(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    camera = Camera(config["camera"])
    detector = Detector(
        config["model"],
        list(config["dataset"]["classes"]),
        float(config["inference"]["confidence_threshold"]),
        list(config["inference"]["roi"]),
    )
    renderer = Renderer(
        config["display"],
        list(config["dataset"]["classes"]),
        list(config["inference"]["roi"]),
    )
    benchmark = Benchmark(float(config["benchmark"]["duration_seconds"]))
    run_dir = _run_dir(config)
    checkpoint_audit: dict[str, Any] = {}
    camera_properties: dict[str, Any] = {}
    warmup_camera_loss = {"skipped_frames": 0, "failed_reads": 0}
    capture_start: dict[str, float | int] = {}
    capture_end: dict[str, float | int] = {}
    stop_reason = "duration"
    runtime_error: Exception | None = None
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")

    try:
        camera.open()
        camera_properties = camera.properties()
        print(
            "Camera: "
            f"{camera_properties.get('width')}x{camera_properties.get('height')} "
            f"@ {camera_properties.get('fps'):.1f} FPS"
        )
        print("Loading fine-tuned RF-DETR checkpoint...")
        checkpoint_audit = detector.load()
        print(
            "Checkpoint mapping: "
            + ", ".join(config["dataset"]["classes"])
        )
        warmup_camera_loss = _warmup(
            camera, detector, int(config["benchmark"]["warmup_frames"])
        )
        capture_start = camera.counters()
        benchmark.start()
        print(
            f"Benchmark started for {config['benchmark']['duration_seconds']:.0f}s. "
            "Press Q to stop."
        )

        while not benchmark.should_stop():
            camera_result = camera.snapshot()
            benchmark.record_camera_loss(
                skipped_frames=camera_result.skipped_frames,
                failed_reads=camera_result.failed_reads,
            )
            if camera_result.frame is None:
                continue

            detection = detector.predict(camera_result.frame)
            processing_fps, previous_frame_age = benchmark.overlay_metrics()
            rendered, render_ms = renderer.render(
                camera_result.frame,
                detection.predictions,
                capture_fps=camera_result.capture_fps,
                processing_fps=processing_fps,
                inference_ms=detection.inference_ms,
                frame_age_ms=previous_frame_age,
            )
            key, display_ms = renderer.show(rendered)
            frame_age_ms = (
                time.perf_counter() - camera_result.captured_at
            ) * 1000.0
            benchmark.record(
                camera_frame_id=camera_result.frame_id,
                capture_fps=camera_result.capture_fps,
                frame_copy_ms=camera_result.frame_copy_ms,
                camera_capture_ms=camera_result.camera_capture_ms,
                frame_age_ms=frame_age_ms,
                inference_ms=detection.inference_ms,
                render_ms=render_ms,
                display_ms=display_ms,
                skipped_frames=camera_result.skipped_frames,
                raw_detections=detection.raw_detections,
                roi_kept_detections=detection.roi_kept_detections,
                roi_rejected_detections=detection.roi_rejected_detections,
            )
            if key in {ord("q"), ord("Q")}:
                stop_reason = "user_q"
                break
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    except Exception as error:
        stop_reason = f"error:{type(error).__name__}"
        runtime_error = error
    finally:
        benchmark.finished_at = (
            time.perf_counter() if benchmark.started_at else 0.0
        )
        if benchmark.started_at:
            capture_end = camera.counters()
        camera.release()
        renderer.close()
        detector.release()

    if not benchmark.started_at:
        if runtime_error is not None:
            raise runtime_error
        raise RuntimeError("Realtime benchmark did not start.")
    metadata = {
        "run": {
            "id": run_dir.name,
            "directory": str(run_dir),
            "started_at": started_at,
        },
        "scenario": {
            "resolution": (
                f"{camera_properties.get('width')}x"
                f"{camera_properties.get('height')}"
            ),
            "confidence_threshold": float(
                config["inference"]["confidence_threshold"]
            ),
            "roi": list(config["inference"]["roi"]),
            "device": str(config["model"]["device"]),
            "precision": str(config["model"].get("precision", "float16")),
            "camera_pipeline": "async_latest_frame",
            "raw_detections": "above_confidence_before_roi",
        },
        "config": config,
        "classes": list(config["dataset"]["classes"]),
        "checkpoint_audit": checkpoint_audit,
        "camera": {
            "requested": config["camera"],
            "actual": camera_properties,
        },
        "warmup": {
            "frames": int(config["benchmark"]["warmup_frames"]),
            **warmup_camera_loss,
        },
        "error": (
            {
                "type": type(runtime_error).__name__,
                "message": str(runtime_error),
            }
            if runtime_error is not None
            else None
        ),
    }
    measured_capture_fps = (
        _capture_fps(capture_start, capture_end)
        if capture_start and capture_end
        else 0.0
    )
    metadata["camera"]["diagnostics"] = _capture_diagnostics(
        camera_properties,
        measured_capture_fps,
        float(config["camera"].get("target_fps", 30)),
    )
    summary = benchmark.write(
        run_dir,
        stop_reason=stop_reason,
        metadata=metadata,
        capture_fps=measured_capture_fps,
    )
    if runtime_error is not None:
        raise runtime_error
    return run_dir, summary


def run_camera_only(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    camera = Camera(config["camera"])
    benchmark = Benchmark(float(config["benchmark"]["duration_seconds"]))
    run_dir = _run_dir(config, "camera")
    camera_properties: dict[str, Any] = {}
    warmup_camera_loss = {"skipped_frames": 0, "failed_reads": 0}
    capture_start: dict[str, float | int] = {}
    capture_end: dict[str, float | int] = {}
    stop_reason = "duration"
    runtime_error: Exception | None = None
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")

    try:
        camera.open()
        camera_properties = camera.properties()
        print(
            "Camera-only benchmark: "
            f"{camera_properties.get('width')}x{camera_properties.get('height')} "
            f"@ requested {config['camera'].get('target_fps')} FPS, "
            f"codec {camera_properties.get('actual_codec') or 'unknown'}"
        )
        warmup_camera_loss = _warmup(
            camera, None, int(config["benchmark"]["warmup_frames"])
        )
        capture_start = camera.counters()
        benchmark.start()
        while not benchmark.should_stop():
            snapshot = camera.snapshot()
            benchmark.record_camera_loss(
                skipped_frames=snapshot.skipped_frames,
                failed_reads=snapshot.failed_reads,
            )
            if snapshot.frame is None:
                continue
            frame_age_ms = (
                time.perf_counter() - snapshot.captured_at
            ) * 1000.0
            benchmark.record(
                camera_frame_id=snapshot.frame_id,
                capture_fps=snapshot.capture_fps,
                frame_copy_ms=snapshot.frame_copy_ms,
                camera_capture_ms=snapshot.camera_capture_ms,
                frame_age_ms=frame_age_ms,
                inference_ms=0.0,
                render_ms=0.0,
                display_ms=0.0,
                skipped_frames=snapshot.skipped_frames,
                raw_detections=0,
                roi_kept_detections=0,
                roi_rejected_detections=0,
            )
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    except Exception as error:
        stop_reason = f"error:{type(error).__name__}"
        runtime_error = error
    finally:
        benchmark.finished_at = (
            time.perf_counter() if benchmark.started_at else 0.0
        )
        if benchmark.started_at:
            capture_end = camera.counters()
        camera.release()

    if not benchmark.started_at:
        if runtime_error is not None:
            raise runtime_error
        raise RuntimeError("Camera-only benchmark did not start.")
    measured_capture_fps = (
        _capture_fps(capture_start, capture_end)
        if capture_start and capture_end
        else 0.0
    )
    target_fps = float(config["camera"].get("target_fps", 30))
    metadata = {
        "run": {
            "id": run_dir.name,
            "directory": str(run_dir),
            "started_at": started_at,
        },
        "scenario": {
            "mode": "camera_only",
            "resolution": (
                f"{camera_properties.get('width')}x"
                f"{camera_properties.get('height')}"
            ),
            "target_fps": target_fps,
        },
        "camera": {
            "requested": config["camera"],
            "actual": camera_properties,
        },
        "warmup": {
            "frames": int(config["benchmark"]["warmup_frames"]),
            **warmup_camera_loss,
        },
        "capture_diagnostics": _capture_diagnostics(
            camera_properties,
            measured_capture_fps,
            target_fps,
        ),
        "error": (
            {
                "type": type(runtime_error).__name__,
                "message": str(runtime_error),
            }
            if runtime_error is not None
            else None
        ),
    }
    summary = benchmark.write(
        run_dir,
        stop_reason=stop_reason,
        metadata=metadata,
        capture_fps=measured_capture_fps,
    )
    if runtime_error is not None:
        raise runtime_error
    return run_dir, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--camera-only",
        action="store_true",
        help="Benchmark camera capture without loading RF-DETR.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_config(
            args.config,
            require_model_checkpoint=not args.camera_only,
        )
        run_dir, summary = (
            run_camera_only(config) if args.camera_only else run(config)
        )
    except Exception:
        traceback.print_exc()
        return 1
    print(f"RUN={run_dir}")
    latency_p95 = summary["frame_age_ms"]["p95"]
    latency_p95_text = (
        f"{latency_p95:.2f}" if latency_p95 is not None else "n/a"
    )
    print(
        f"CAPTURE_FPS={summary['capture_fps']:.2f} "
        f"PROCESSING_FPS={summary['processing_fps']:.2f} "
        f"FRAME_AGE_P95_MS={latency_p95_text} "
        f"FRAMES={summary['processed_frames']}"
    )
    diagnostics = summary["metadata"].get("capture_diagnostics")
    if diagnostics is None:
        diagnostics = summary["metadata"].get("camera", {}).get(
            "diagnostics", []
        )
    for diagnostic in diagnostics:
        print(f"CAMERA_WARNING={diagnostic}")
    return 0


if __name__ == "__main__":
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if callable(reconfigure):
                reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
