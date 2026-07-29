"""Asynchronous OpenCV camera capture that always returns the latest frame."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from threading import Condition, Event, Thread, current_thread
from typing import Any

import cv2
import numpy as np


@dataclass
class CameraRead:
    frame: np.ndarray[Any, Any] | None
    read_ms: float
    dropped_reads: int
    frame_id: int = 0
    captured_at: float = 0.0


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
        self._condition = Condition()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._latest_frame: np.ndarray[Any, Any] | None = None
        self._latest_frame_id = 0
        self._delivered_frame_id = 0
        self._latest_read_ms = 0.0
        self._latest_captured_at = 0.0
        self._failed_reads = 0
        self._reported_failed_reads = 0
        self._error: Exception | None = None
        self._properties: dict[str, Any] = {"opened": False}

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
            self.capture.release()
            self.capture = None
            raise ConnectionError(
                f"Cannot open camera {self.camera_id} with backend "
                f"{self.backend_name!r}."
            )
        self._properties = self._capture_properties(self.capture)
        with self._condition:
            self._latest_frame = None
            self._latest_frame_id = 0
            self._delivered_frame_id = 0
            self._latest_read_ms = 0.0
            self._latest_captured_at = 0.0
            self._failed_reads = 0
            self._reported_failed_reads = 0
            self._error = None
        self._stop_event.clear()
        self._thread = Thread(
            target=self._capture_loop,
            name=f"camera-{self.camera_id}-latest-frame",
            daemon=True,
        )
        self._thread.start()

    def _capture_properties(
        self, capture: cv2.VideoCapture
    ) -> dict[str, Any]:
        return {
            "opened": bool(capture.isOpened()),
            "id": self.camera_id,
            "backend": self.backend_name,
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "mode": "async_latest_frame",
        }

    def _reconnect(self) -> bool:
        attempts = int(self.config.get("reconnect_attempts", 3))
        delay = float(self.config.get("reconnect_delay_seconds", 0.25))
        for _ in range(attempts):
            if self._stop_event.is_set():
                return False
            if self.capture is not None:
                self.capture.release()
                self.capture = None
            if self._stop_event.wait(delay):
                return False
            capture = self._new_capture()
            if capture.isOpened():
                self.capture = capture
                self._properties = self._capture_properties(capture)
                return True
            capture.release()
        return False

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            capture = self.capture
            if capture is None or not capture.isOpened():
                if not self._reconnect():
                    self._set_error(
                        ConnectionError(
                            f"Camera {self.camera_id} could not reconnect."
                        )
                    )
                    return
                continue

            captured_at = time.perf_counter()
            ok, frame = capture.read()
            read_ms = (time.perf_counter() - captured_at) * 1000.0
            if ok and frame is not None and frame.size:
                with self._condition:
                    self._latest_frame = frame
                    self._latest_frame_id += 1
                    self._latest_read_ms = read_ms
                    self._latest_captured_at = captured_at
                    self._condition.notify_all()
                continue

            with self._condition:
                self._failed_reads += 1
            if not self._reconnect():
                self._set_error(
                    ConnectionError(
                        f"Camera {self.camera_id} stopped returning frames."
                    )
                )
                return

    def _set_error(self, error: Exception) -> None:
        with self._condition:
            self._error = error
            self._condition.notify_all()

    def read(self) -> CameraRead:
        target_fps = max(1.0, float(self.config.get("target_fps", 30)))
        timeout = float(
            self.config.get("read_timeout_seconds", max(1.0, 3 / target_fps))
        )
        deadline = time.perf_counter() + timeout
        with self._condition:
            while (
                self._latest_frame_id <= self._delivered_frame_id
                and self._error is None
                and not self._stop_event.is_set()
            ):
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)

            failed = self._failed_reads - self._reported_failed_reads
            self._reported_failed_reads = self._failed_reads
            if self._error is not None:
                raise ConnectionError(str(self._error)) from self._error
            if self._latest_frame_id <= self._delivered_frame_id:
                return CameraRead(None, 0.0, failed)

            skipped = max(
                0, self._latest_frame_id - self._delivered_frame_id - 1
            )
            self._delivered_frame_id = self._latest_frame_id
            return CameraRead(
                frame=self._latest_frame,
                read_ms=self._latest_read_ms,
                dropped_reads=failed + skipped,
                frame_id=self._latest_frame_id,
                captured_at=self._latest_captured_at,
            )

    def properties(self) -> dict[str, Any]:
        return dict(self._properties)

    def release(self) -> None:
        self._stop_event.set()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not current_thread():
            thread.join(timeout=2.0)
        self._thread = None
        self._properties = {"opened": False}
        with self._condition:
            self._condition.notify_all()
