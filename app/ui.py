from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from .config_loader import OverlayConfig
from .inspection import InspectionResult


@dataclass
class UIState:
    status: str
    sub_status: Optional[str]
    color: tuple[int, int, int]


class Display:
    def __init__(self, cfg: OverlayConfig, window_name: str = "Inspection"):
        self._cfg = cfg
        self._window_name = window_name
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            self._window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
        self._add_part_handler: Optional[Callable[[], None]] = None
        self._add_rect = (16, 16, 32, 32)  # x, y, size, size
        cv2.setMouseCallback(self._window_name, self._handle_mouse)

    def show_idle(self, frame: np.ndarray, is_stable: bool, debug: Optional[str] = None) -> None:
        canvas = frame.copy()
        self._draw_add_button(canvas)
        status = "Hold part steady" if not is_stable else "Capturing..."
        color = (50, 150, 255) if not is_stable else (0, 200, 0)
        self._overlay_text(canvas, status, color, debug)
        cv2.imshow(self._window_name, canvas)

    def show_result(self, result: InspectionResult, banner: str) -> None:
        canvas = result.annotated
        self._draw_add_button(canvas)
        color = (0, 255, 0) if result.passed else (0, 0, 255)
        reasons = "\n".join(result.reasons[:3]) if result.reasons else ""
        self._overlay_text(canvas, banner, color, reasons)
        cv2.imshow(self._window_name, canvas)

    def close(self) -> None:
        cv2.destroyWindow(self._window_name)

    def _overlay_text(
        self,
        frame: np.ndarray,
        text: str,
        color: tuple[int, int, int],
        extra: Optional[str] = None,
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self._cfg.font_scale
        thickness = self._cfg.thickness
        org = (40, 80)
        cv2.putText(frame, text, org, font, font_scale * 1.6, color, thickness + 1, cv2.LINE_AA)
        if extra:
            for i, line in enumerate(extra.splitlines()):
                cv2.putText(
                    frame,
                    line,
                    (org[0], org[1] + 40 * (i + 1)),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

    def set_add_part_handler(self, handler: Callable[[], None]) -> None:
        self._add_part_handler = handler

    def _draw_add_button(self, frame: np.ndarray) -> None:
        x, y, w, h = self._add_rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
        cv2.line(frame, (x + w // 2, y + 4), (x + w // 2, y + h - 4), (255, 255, 255), 2)
        cv2.line(frame, (x + 4, y + h // 2), (x + w - 4, y + h // 2), (255, 255, 255), 2)

    def _handle_mouse(self, event: int, x: int, y: int, _flags: int, _userdata) -> None:
        if event != cv2.EVENT_LBUTTONUP:
            return
        if self._add_part_handler is None:
            return
        if self._point_in_add_button(x, y):
            self._add_part_handler()

    def _point_in_add_button(self, x: int, y: int) -> bool:
        bx, by, bw, bh = self._add_rect
        return bx <= x <= bx + bw and by <= y <= by + bh
