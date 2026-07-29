"""Load one fine-tuned RF-DETR checkpoint and infer camera frames."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from ..train import (
    checkpoint_audit,
    detections_to_predictions,
    load_best_model,
)


@dataclass
class DetectionResult:
    predictions: list[dict[str, Any]]
    inference_ms: float


class Detector:
    def __init__(
        self,
        model_config: dict[str, Any],
        classes: list[str],
        confidence_threshold: float,
        roi: list[float] | None = None,
    ) -> None:
        self.config = model_config
        self.classes = classes
        self.threshold = float(confidence_threshold)
        self.roi = list(roi or [0.0, 0.0, 1.0, 1.0])
        self.model: Any = None
        self.audit: dict[str, Any] = {}

    def load(self) -> dict[str, Any]:
        import torch

        checkpoint = Path(self.config["checkpoint"])
        expected_count = int(
            self.config.get("expected_class_count", len(self.classes))
        )
        if len(self.classes) != expected_count:
            raise ValueError(
                f"Expected {expected_count} classes, got {len(self.classes)}: "
                f"{self.classes}"
            )
        device = str(self.config.get("device", "cuda")).lower()
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        self.audit = checkpoint_audit(checkpoint, self.classes)
        self.model = load_best_model(
            checkpoint, str(self.config["name"]), self.classes
        )
        requested_device = torch.device(device)
        self.model.model.device = requested_device
        if bool(self.config.get("optimize", True)):
            precision_name = str(
                self.config.get("precision", "float16")
            ).lower()
            precision = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }.get(precision_name)
            if precision is None:
                raise ValueError(f"Unsupported inference precision: {precision_name}")
            if requested_device.type == "cpu" and precision is not torch.float32:
                raise ValueError("CPU realtime inference requires precision=float32.")
            self.model.inference(
                compile=bool(self.config.get("compile", False)),
                batch_size=1,
                dtype=precision,
                inplace=True,
            )
        self.audit["runtime"] = {
            "device": str(requested_device),
            "optimized": bool(self.config.get("optimize", True)),
            "compile": bool(self.config.get("compile", False)),
            "precision": str(self.config.get("precision", "float16")),
        }
        return self.audit

    def predict(self, frame_bgr: Any) -> DetectionResult:
        if self.model is None:
            raise RuntimeError("Detector.load() must be called before predict().")
        import torch

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        detections = self.model.predict(
            frame_rgb,
            threshold=self.threshold,
            include_source_image=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - started) * 1000.0
        predictions = [
            prediction
            for prediction in detections_to_predictions(detections, self.classes)
            if float(prediction["score"]) >= self.threshold
        ]
        height, width = frame_bgr.shape[:2]
        roi_x, roi_y, roi_width, roi_height = self.roi
        left = roi_x * width
        top = roi_y * height
        right = (roi_x + roi_width) * width
        bottom = (roi_y + roi_height) * height
        predictions = [
            prediction
            for prediction in predictions
            if left
            <= (float(prediction["xyxy"][0]) + float(prediction["xyxy"][2])) / 2.0
            <= right
            and top
            <= (float(prediction["xyxy"][1]) + float(prediction["xyxy"][3])) / 2.0
            <= bottom
        ]
        return DetectionResult(predictions, inference_ms)

    def release(self) -> None:
        self.model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
