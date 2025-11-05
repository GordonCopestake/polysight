from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config_loader import GeometryConfig


class GeometryError(RuntimeError):
    pass


@dataclass
class GeometryState:
    mm_per_px: float
    source: str


class GeometryResolver:
    def __init__(self, cfg: GeometryConfig):
        self._cfg = cfg
        self._cached: Optional[GeometryState] = None

    def resolve(self, frame: np.ndarray) -> GeometryState:
        if self._cached is not None:
            return self._cached
        if self._cfg.mm_per_px_override:
            self._cached = GeometryState(
                mm_per_px=float(self._cfg.mm_per_px_override),
                source="override",
            )
            return self._cached
        if not self._cfg.fiducial:
            raise GeometryError("No geometry override or fiducial strategy configured")
        state = self._resolve_from_fiducial(frame)
        self._cached = state
        return state

    def _resolve_from_fiducial(self, frame: np.ndarray) -> GeometryState:
        fid = self._cfg.fiducial
        if fid is None:
            raise GeometryError("Fiducial configuration missing")

        try:
            dictionary = getattr(cv2.aruco, fid.aruco_dict)
        except AttributeError as exc:
            raise GeometryError(f"Unknown ArUco dictionary {fid.aruco_dict}") from exc

        aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            raise GeometryError("No ArUco markers detected for fiducial scaling")

        required_ids = set(fid.ids)
        observed = {int(_id): corner for _id, corner in zip(ids.flatten(), corners)}
        intersect = required_ids.intersection(observed.keys())
        if not intersect:
            raise GeometryError(
                f"ArUco markers {fid.ids} not found; observed {sorted(observed.keys())}"
            )

        mm_per_px_values = []
        for marker_id in intersect:
            corner = observed[marker_id]
            points = np.squeeze(corner)
            if points.ndim != 2 or points.shape[0] < 4:
                continue
            side_lengths = []
            for i in range(4):
                p1 = points[i % points.shape[0]]
                p2 = points[(i + 1) % points.shape[0]]
                side_lengths.append(np.linalg.norm(p1 - p2))
            avg_side_px = float(np.mean(side_lengths))
            if avg_side_px <= 0:
                continue
            mm_per_px_values.append(fid.tag_size_mm / avg_side_px)

        if not mm_per_px_values:
            raise GeometryError("Failed to compute mm/px from detected markers")

        mm_per_px = float(np.mean(mm_per_px_values))
        return GeometryState(mm_per_px=mm_per_px, source="fiducial")
