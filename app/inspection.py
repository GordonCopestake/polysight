from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import List, Optional

import cv2
import numpy as np

from .config_loader import HoleConfig, InspectionConfig, SilhouetteConfig
from .geometry import GeometryState
from .utils import Rect


@dataclass
class HoleMeasurement:
    name: str
    diameter_mm: Optional[float]
    diameter_error_mm: Optional[float]
    center_offset_mm: Optional[float]
    passed: bool
    reason: Optional[str]
    contour: Optional[np.ndarray]
    roi: Rect


@dataclass
class SilhouetteMeasurement:
    mismatch_pct: float
    passed: bool
    reason: Optional[str]
    roi: Rect


@dataclass
class InspectionResult:
    passed: bool
    reasons: List[str]
    holes: List[HoleMeasurement]
    silhouette: Optional[SilhouetteMeasurement]
    annotated: np.ndarray


class Inspector:
    def __init__(self, cfg: InspectionConfig):
        self._cfg = cfg
        self._silhouette_template: Optional[np.ndarray] = None
        if cfg.silhouette and cfg.silhouette.reference_path:
            self._silhouette_template = self._load_template(cfg.silhouette)

    def inspect(self, frame: np.ndarray, geom: GeometryState) -> InspectionResult:
        annotated = frame.copy()
        reasons: List[str] = []

        hole_results = [self._measure_hole(frame, hole, geom, annotated) for hole in self._cfg.holes]
        for hr in hole_results:
            if not hr.passed and hr.reason:
                reasons.append(f"{hr.name}: {hr.reason}")

        silhouette_result = None
        if self._cfg.silhouette and self._cfg.silhouette.enabled:
            silhouette_result = self._check_silhouette(frame, self._cfg.silhouette, annotated)
            if silhouette_result and not silhouette_result.passed and silhouette_result.reason:
                reasons.append(f"silhouette: {silhouette_result.reason}")

        return InspectionResult(
            passed=len(reasons) == 0,
            reasons=reasons,
            holes=hole_results,
            silhouette=silhouette_result,
            annotated=annotated,
        )

    def _measure_hole(
        self,
        frame: np.ndarray,
        hole: HoleConfig,
        geom: GeometryState,
        annotated: np.ndarray,
    ) -> HoleMeasurement:
        roi = hole.roi
        sub_img = frame[roi.as_slice()].copy()
        if sub_img.size == 0:
            reason = "ROI is empty"
            self._draw_roi(annotated, roi, (0, 0, 255), reason)
            return HoleMeasurement(hole.name, None, None, None, False, reason, None, roi)

        gray = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            reason = "no contour found"
            self._draw_roi(annotated, roi, (0, 0, 255), reason)
            return HoleMeasurement(hole.name, None, None, None, False, reason, None, roi)

        contour = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        diameter_px = radius * 2.0
        diameter_mm = diameter_px * geom.mm_per_px
        diameter_error = abs(diameter_mm - hole.nominal_diameter_mm)

        m = cv2.moments(contour)
        if m["m00"] > 0:
            mx = float(m["m10"] / m["m00"])
            my = float(m["m01"] / m["m00"])
        else:
            mx, my = cx, cy
        offset_px = hypot(mx - roi.width / 2, my - roi.height / 2)
        offset_mm = offset_px * geom.mm_per_px

        passed = True
        reason = None
        if diameter_error > hole.diameter_tolerance_mm:
            passed = False
            reason = f"diameter {diameter_mm:.2f}mm off by {diameter_error:.2f}mm"
        if offset_mm > hole.center_tolerance_mm:
            passed = False
            extra = f"center offset {offset_mm:.2f}mm"
            reason = f"{reason}; {extra}" if reason else extra

        self._draw_hole_annotation(annotated, roi, (cx, cy), radius, passed)

        return HoleMeasurement(
            name=hole.name,
            diameter_mm=diameter_mm,
            diameter_error_mm=diameter_error,
            center_offset_mm=offset_mm,
            passed=passed,
            reason=reason,
            contour=contour,
            roi=roi,
        )

    def _check_silhouette(
        self, frame: np.ndarray, cfg: SilhouetteConfig, annotated: np.ndarray
    ) -> Optional[SilhouetteMeasurement]:
        roi = cfg.roi
        sub_img = frame[roi.as_slice()].copy()
        if sub_img.size == 0:
            reason = "ROI is empty"
            self._draw_roi(annotated, roi, (0, 0, 255), reason)
            return SilhouetteMeasurement(0.0, False, reason, roi)

        gray = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, live_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if self._silhouette_template is None:
            self._silhouette_template = live_mask
            reason = "baseline captured"
            self._draw_roi(annotated, roi, (0, 255, 255), reason)
            return SilhouetteMeasurement(0.0, True, reason, roi)

        if self._silhouette_template.shape != live_mask.shape:
            self._silhouette_template = cv2.resize(
                self._silhouette_template, (live_mask.shape[1], live_mask.shape[0])
            )

        diff = cv2.absdiff(self._silhouette_template, live_mask)
        mismatch = np.count_nonzero(diff > cfg.diff_threshold)
        total = diff.size
        mismatch_pct = (mismatch / total) * 100.0 if total else 100.0

        passed = mismatch_pct <= cfg.max_mismatch_pct
        reason = None if passed else f"mismatch {mismatch_pct:.2f}%"
        color = (0, 200, 0) if passed else (0, 0, 255)
        self._draw_roi(annotated, roi, color, reason)

        return SilhouetteMeasurement(mismatch_pct, passed, reason, roi)

    @staticmethod
    def _draw_roi(image: np.ndarray, roi: Rect, color: tuple[int, int, int], label: Optional[str]) -> None:
        top_left = (roi.x, roi.y)
        bottom_right = (roi.x + roi.width, roi.y + roi.height)
        cv2.rectangle(image, top_left, bottom_right, color, 2)
        if label:
            cv2.putText(
                image,
                label,
                (roi.x, roi.y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _draw_hole_annotation(
        image: np.ndarray,
        roi: Rect,
        center: tuple[float, float],
        radius: float,
        passed: bool,
    ) -> None:
        cx, cy = int(roi.x + center[0]), int(roi.y + center[1])
        rad = int(radius)
        color = (0, 200, 0) if passed else (0, 0, 255)
        cv2.circle(image, (cx, cy), rad, color, 2)
        cv2.circle(image, (cx, cy), 2, color, -1)

    @staticmethod
    def _load_template(cfg: SilhouetteConfig) -> Optional[np.ndarray]:
        if not cfg.reference_path or not cfg.reference_path.exists():
            return None
        template = cv2.imread(str(cfg.reference_path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise FileNotFoundError(f"Unable to read silhouette reference {cfg.reference_path}")
        _, mask = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask
