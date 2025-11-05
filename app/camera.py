from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Tuple

import cv2
import numpy as np
import platform

from .config_loader import CaptureConfig


@dataclass
class Frame:
    image: np.ndarray
    timestamp_s: float


class CameraError(RuntimeError):
    pass


class Camera:
    def __init__(self, camera_id: int, capture_cfg: CaptureConfig):
        self._camera_id = camera_id
        self._cfg = capture_cfg
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self._camera_id, backend)
        if not self._cap.isOpened():
            raise CameraError(f"Unable to open camera {self._camera_id}")
        width, height = self._cfg.resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, self._cfg.fps)

        for _ in range(max(1, self._cfg.warmup_frames)):
            _, _ = self._cap.read()
            time.sleep(0.01)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self) -> Frame:
        if self._cap is None:
            raise CameraError("Camera is not opened")
        ok, frame = self._cap.read()
        if not ok:
            raise CameraError("Failed to read frame from camera")
        return Frame(image=frame, timestamp_s=time.time())

    def resolution(self) -> Tuple[int, int]:
        return self._cfg.resolution

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


@contextmanager
def open_camera(camera_id: int, capture_cfg: CaptureConfig) -> Iterator[Camera]:
    cam = Camera(camera_id, capture_cfg)
    try:
        cam.open()
        yield cam
    finally:
        cam.close()
