from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .camera import Camera, open_camera
from .config_loader import Config
from .geometry import GeometryError, GeometryResolver, GeometryState
from .inspection import Inspector, InspectionResult
from .storage import Storage
from .ui import Display
from .utils import Rect


@dataclass
class PipelineState:
    stable_counter: int = 0
    hold_counter: int = 0
    last_frame: Optional[np.ndarray] = None
    last_result: Optional[InspectionResult] = None
    geometry: Optional[GeometryState] = None


class Pipeline:
    def __init__(self, cfg: Config):
        self._cfg = cfg
        self._display = Display(cfg.overlay)
        self._storage = Storage(cfg.storage, cfg.sku)
        self._inspector = Inspector(cfg.inspection)
        self._geometry_resolver = GeometryResolver(cfg.geometry)

    def run(self) -> None:
        with open_camera(self._cfg.camera_id, self._cfg.capture) as camera:
            state = PipelineState()
            self._loop(camera, state)
        self._display.close()

    def _loop(self, camera: Camera, state: PipelineState) -> None:
        while True:
            frame = camera.read().image
            is_stable, jitter = self._is_stable(frame, state)

            if state.geometry is None:
                try:
                    state.geometry = self._geometry_resolver.resolve(frame)
                except GeometryError as exc:
                    debug_text = f"Geometry: {exc}"
                    self._display.show_idle(frame, False, debug_text)
                    if self._should_quit():
                        break
                    state.last_frame = frame
                    continue

            if state.last_result is None:
                if is_stable:
                    state.stable_counter += 1
                else:
                    state.stable_counter = 0

                if state.stable_counter >= self._cfg.stabilization.required_frames:
                    result = self._inspector.inspect(frame.copy(), state.geometry)
                    self._storage.save(result, frame)
                    banner = "GREEN" if result.passed else "RED"
                    self._display.show_result(result, banner)
                    state.last_result = result
                    state.hold_counter = 0
                else:
                    debug = f"jitter {jitter:.2f}" if self._cfg.overlay.show_debug else None
                    self._display.show_idle(frame, is_stable, debug)
            else:
                state.hold_counter += 1
                hold_frames = int(self._cfg.capture.fps * 1.5)
                banner = "GREEN" if state.last_result.passed else "RED"
                self._display.show_result(state.last_result, banner)
                if not is_stable or state.hold_counter > hold_frames:
                    if not is_stable:
                        self._storage.reset_run()
                    state.last_result = None
                    state.stable_counter = 0

            state.last_frame = frame
            if self._should_quit():
                break

    def _is_stable(self, frame: np.ndarray, state: PipelineState) -> tuple[bool, float]:
        if state.last_frame is None:
            return False, 0.0
        roi = self._cfg.stabilization.roi
        diff = self._roi_diff(frame, state.last_frame, roi)
        threshold = self._cfg.stabilization.pixel_jitter_threshold
        return diff < threshold, diff

    @staticmethod
    def _roi_diff(current: np.ndarray, previous: np.ndarray, roi: Rect) -> float:
        cur = current[roi.as_slice()]
        prev = previous[roi.as_slice()]
        if cur.size == 0 or prev.size == 0:
            return 0.0
        cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(cur_gray, prev_gray)
        return float(np.mean(diff))

    @staticmethod
    def _should_quit() -> bool:
        key = cv2.waitKey(1) & 0xFF
        return key in (27, ord("q"))
