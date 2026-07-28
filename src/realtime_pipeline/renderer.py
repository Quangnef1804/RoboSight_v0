"""Render RF-DETR detections and realtime performance overlays."""

from __future__ import annotations

import colorsys
import time
from typing import Any

import cv2
import numpy as np


def class_colors(classes: list[str]) -> dict[str, tuple[int, int, int]]:
    named = {
        "black": (180, 180, 180),
        "green": (0, 220, 0),
        "purple": (220, 0, 220),
        "red": (0, 0, 255),
        "yellow": (0, 230, 255),
    }
    colors: dict[str, tuple[int, int, int]] = {}
    for index, class_name in enumerate(classes):
        if class_name.lower() in named:
            colors[class_name] = named[class_name.lower()]
            continue
        hue = index / max(1, len(classes))
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
        colors[class_name] = (
            round(blue * 255),
            round(green * 255),
            round(red * 255),
        )
    return colors


class Renderer:
    def __init__(
        self, config: dict[str, Any], classes: list[str]
    ) -> None:
        self.config = config
        self.enabled = bool(config.get("enabled", True))
        self.window_name = str(
            config.get("window_name", "RoboSight RF-DETR Realtime")
        )
        self.colors = class_colors(classes)
        self.window_open = False
        if self.enabled:
            gui_line = next(
                (
                    line.strip()
                    for line in cv2.getBuildInformation().splitlines()
                    if line.strip().startswith("GUI:")
                ),
                "",
            )
            if not gui_line or gui_line.upper().endswith("NONE"):
                raise RuntimeError(
                    "Realtime display requires opencv-python with GUI support."
                )

    def render(
        self,
        frame: np.ndarray[Any, Any],
        predictions: list[dict[str, Any]],
        *,
        fps: float,
        inference_ms: float,
        latency_ms: float,
    ) -> tuple[np.ndarray[Any, Any], float]:
        started = time.perf_counter()
        rendered = frame.copy()
        for prediction in predictions:
            x1, y1, x2, y2 = (
                round(float(value)) for value in prediction["xyxy"]
            )
            class_name = str(prediction["class_name"])
            score = float(prediction["score"])
            color = self.colors[class_name]
            cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {score:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            top = max(0, y1 - text_height - baseline - 8)
            cv2.rectangle(
                rendered,
                (x1, top),
                (x1 + text_width + 8, top + text_height + baseline + 8),
                color,
                -1,
            )
            cv2.putText(
                rendered,
                label,
                (x1 + 4, top + text_height + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
        status = (
            f"FPS {fps:5.1f} | inference {inference_ms:6.1f} ms | "
            f"latency {latency_ms:6.1f} ms | objects {len(predictions)}"
        )
        cv2.rectangle(rendered, (0, 0), (rendered.shape[1], 42), (20, 20, 20), -1)
        cv2.putText(
            rendered,
            status,
            (12, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return rendered, (time.perf_counter() - started) * 1000.0

    def show(self, frame: np.ndarray[Any, Any]) -> tuple[int, float]:
        if not self.enabled:
            return -1, 0.0
        started = time.perf_counter()
        if self.window_open:
            try:
                if cv2.getWindowProperty(
                    self.window_name, cv2.WND_PROP_VISIBLE
                ) < 1:
                    return ord("q"), (time.perf_counter() - started) * 1000.0
            except cv2.error:
                return ord("q"), (time.perf_counter() - started) * 1000.0
        if not self.window_open:
            cv2.namedWindow(
                self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
            )
            height, width = frame.shape[:2]
            scale = min(
                1.0,
                int(self.config.get("max_width", 1280)) / width,
                int(self.config.get("max_height", 720)) / height,
            )
            cv2.resizeWindow(
                self.window_name, round(width * scale), round(height * scale)
            )
            self.window_open = True
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key, (time.perf_counter() - started) * 1000.0

    def close(self) -> None:
        if self.window_open:
            cv2.destroyWindow(self.window_name)
            self.window_open = False
