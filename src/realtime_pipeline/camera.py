"""OpenCV camera capture with bounded reconnect handling."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class CameraRead:
    frame: np.ndarray[Any, Any] | None
    read_ms: float
    dropped_reads: int


class Camera:
    BACKENDS = {
        "any": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "v4l2": cv2.CAP_V4L2,
    }

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.camera_id = int(config["id"])
        backend_name = str(config.get("backend", "any")).lower()
        if backend_name not in self.BACKENDS:
            raise ValueError(f"Unsupported camera backend: {backend_name}")
        if backend_name == "dshow" and sys.platform != "win32":
            backend_name = "any"
        self.backend_name = backend_name
        self.backend = self.BACKENDS[backend_name]
        self.capture: cv2.VideoCapture | None = None

    def _new_capture(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self.camera_id, self.backend)
        resolution = self.config["resolution"]
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(resolution["width"]))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(resolution["height"]))
        capture.set(cv2.CAP_PROP_FPS, float(self.config.get("target_fps", 30)))
        capture.set(
            cv2.CAP_PROP_BUFFERSIZE, int(self.config.get("buffer_size", 1))
        )
        return capture

    def open(self) -> None:
        self.release()
        self.capture = self._new_capture()
        if not self.capture.isOpened():
            self.release()
            raise ConnectionError(
                f"Cannot open camera {self.camera_id} with backend "
                f"{self.backend_name!r}."
            )

    def _reconnect(self) -> bool:
        self.release()
        time.sleep(float(self.config.get("reconnect_delay_seconds", 0.25)))
        self.capture = self._new_capture()
        return bool(self.capture.isOpened())

    def read(self) -> CameraRead:
        started = time.perf_counter()
        dropped = 0
        attempts = int(self.config.get("reconnect_attempts", 3)) + 1
        for attempt in range(attempts):
            if self.capture is None or not self.capture.isOpened():
                if not self._reconnect():
                    dropped += 1
                    continue
            assert self.capture is not None
            ok, frame = self.capture.read()
            if ok and frame is not None and frame.size:
                return CameraRead(
                    frame=frame,
                    read_ms=(time.perf_counter() - started) * 1000.0,
                    dropped_reads=dropped,
                )
            dropped += 1
            if attempt < attempts - 1:
                self._reconnect()
        return CameraRead(
            frame=None,
            read_ms=(time.perf_counter() - started) * 1000.0,
            dropped_reads=dropped,
        )

    def properties(self) -> dict[str, Any]:
        if self.capture is None or not self.capture.isOpened():
            return {"opened": False}
        return {
            "opened": True,
            "id": self.camera_id,
            "backend": self.backend_name,
            "width": int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(self.capture.get(cv2.CAP_PROP_FPS)),
        }

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
