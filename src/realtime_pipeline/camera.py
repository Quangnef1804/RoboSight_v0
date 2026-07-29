"""Asynchronous OpenCV camera capture that always returns the latest frame."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from threading import Event, Lock, Thread, current_thread
from typing import Any

import cv2
import numpy as np


@dataclass
class CameraSnapshot:
    frame: np.ndarray[Any, Any] | None
    frame_copy_ms: float
    camera_capture_ms: float
    capture_fps: float
    skipped_frames: int
    failed_reads: int
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
        self._lock = Lock()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._latest_frame: np.ndarray[Any, Any] | None = None
        self._latest_frame_id = 0
        self._delivered_frame_id = 0
        self._latest_capture_ms = 0.0
        self._latest_capture_fps = 0.0
        self._latest_captured_at = 0.0
        self._captured_frames = 0
        self._failed_reads = 0
        self._reported_failed_reads = 0
        self._error: Exception | None = None
        self._properties: dict[str, Any] = {"opened": False}
        self._setting_results: dict[str, bool] = {}

    def _new_capture(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self.camera_id, self.backend)
        resolution = self.config["resolution"]
        codec = str(self.config.get("codec", "MJPG")).upper()
        setting_results = {
            "codec": bool(
                capture.set(
                    cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(*codec),
                )
            ),
            "width": bool(
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(resolution["width"]))
            ),
            "height": bool(
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(resolution["height"]))
            ),
            "fps": bool(
                capture.set(
                    cv2.CAP_PROP_FPS,
                    float(self.config.get("target_fps", 30)),
                )
            ),
            "buffer_size": bool(
                capture.set(
                    cv2.CAP_PROP_BUFFERSIZE,
                    int(self.config.get("buffer_size", 1)),
                )
            ),
        }
        if "auto_exposure" in self.config:
            setting_results["auto_exposure"] = bool(
                capture.set(
                    cv2.CAP_PROP_AUTO_EXPOSURE,
                    float(self.config["auto_exposure"]),
                )
            )
        if "exposure" in self.config:
            setting_results["exposure"] = bool(
                capture.set(cv2.CAP_PROP_EXPOSURE, float(self.config["exposure"]))
            )
        self._setting_results = setting_results
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
        with self._lock:
            self._latest_frame = None
            self._latest_frame_id = 0
            self._delivered_frame_id = 0
            self._latest_capture_ms = 0.0
            self._latest_capture_fps = 0.0
            self._latest_captured_at = 0.0
            self._captured_frames = 0
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
        fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(
            chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4)
        ).strip("\x00")
        return {
            "opened": bool(capture.isOpened()),
            "id": self.camera_id,
            "backend": self.backend_name,
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "requested_codec": str(self.config.get("codec", "MJPG")).upper(),
            "actual_codec": fourcc,
            "auto_exposure": float(capture.get(cv2.CAP_PROP_AUTO_EXPOSURE)),
            "exposure": float(capture.get(cv2.CAP_PROP_EXPOSURE)),
            "set_return_values": dict(self._setting_results),
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

            capture_started_at = time.perf_counter()
            ok, frame = capture.read()
            captured_at = time.perf_counter()
            capture_ms = (captured_at - capture_started_at) * 1000.0
            if ok and frame is not None and frame.size:
                with self._lock:
                    previous_captured_at = self._latest_captured_at
                    self._latest_frame = frame
                    self._latest_frame_id += 1
                    self._latest_capture_ms = capture_ms
                    if previous_captured_at > 0:
                        interval = captured_at - previous_captured_at
                        self._latest_capture_fps = (
                            1.0 / interval if interval > 0 else 0.0
                        )
                    self._latest_captured_at = captured_at
                    self._captured_frames += 1
                continue

            with self._lock:
                self._failed_reads += 1
            if not self._reconnect():
                self._set_error(
                    ConnectionError(
                        f"Camera {self.camera_id} stopped returning frames."
                    )
                )
                return

    def _set_error(self, error: Exception) -> None:
        with self._lock:
            self._error = error

    def snapshot(self) -> CameraSnapshot:
        """Copy the newest unprocessed frame without waiting for camera I/O."""
        started = time.perf_counter()
        with self._lock:
            failed = self._failed_reads - self._reported_failed_reads
            self._reported_failed_reads = self._failed_reads
            if self._error is not None:
                raise ConnectionError(str(self._error)) from self._error
            if self._latest_frame_id <= self._delivered_frame_id:
                return CameraSnapshot(
                    frame=None,
                    frame_copy_ms=(time.perf_counter() - started) * 1000.0,
                    camera_capture_ms=0.0,
                    capture_fps=0.0,
                    skipped_frames=0,
                    failed_reads=failed,
                )

            skipped = max(
                0, self._latest_frame_id - self._delivered_frame_id - 1
            )
            self._delivered_frame_id = self._latest_frame_id
            frame = self._latest_frame.copy()
            frame_id = self._latest_frame_id
            captured_at = self._latest_captured_at
            camera_capture_ms = self._latest_capture_ms
            capture_fps = self._latest_capture_fps
        return CameraSnapshot(
            frame=frame,
            frame_copy_ms=(time.perf_counter() - started) * 1000.0,
            camera_capture_ms=camera_capture_ms,
            capture_fps=capture_fps,
            skipped_frames=skipped,
            failed_reads=failed,
            frame_id=frame_id,
            captured_at=captured_at,
        )

    def counters(self) -> dict[str, float | int]:
        with self._lock:
            return {
                "captured_frames": self._captured_frames,
                "failed_reads": self._failed_reads,
                "timestamp": time.perf_counter(),
            }

    def properties(self) -> dict[str, Any]:
        with self._lock:
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
        with self._lock:
            self._properties = {"opened": False}
