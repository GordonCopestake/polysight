from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .utils import Rect


@dataclass
class CaptureConfig:
    fps: int
    resolution: tuple[int, int]
    warmup_frames: int


@dataclass
class FiducialConfig:
    aruco_dict: str
    ids: List[int]
    tag_size_mm: float


@dataclass
class GeometryConfig:
    mm_per_px_override: Optional[float]
    fiducial: Optional[FiducialConfig]


@dataclass
class StabilizationConfig:
    roi: Rect
    required_frames: int
    pixel_jitter_threshold: float


@dataclass
class HoleConfig:
    name: str
    roi: Rect
    nominal_diameter_mm: float
    diameter_tolerance_mm: float
    center_tolerance_mm: float


@dataclass
class SilhouetteConfig:
    enabled: bool
    roi: Rect
    diff_threshold: int
    max_mismatch_pct: float
    reference_path: Optional[Path] = None


@dataclass
class InspectionConfig:
    holes: List[HoleConfig]
    silhouette: Optional[SilhouetteConfig] = None


@dataclass
class OverlayConfig:
    font_scale: float = 0.8
    thickness: int = 2
    show_debug: bool = False


@dataclass
class StorageConfig:
    root: Path
    save_annotated: bool
    save_raw: bool


@dataclass
class Config:
    sku: str
    camera_id: int
    capture: CaptureConfig
    geometry: GeometryConfig
    stabilization: StabilizationConfig
    inspection: InspectionConfig
    overlay: OverlayConfig
    storage: StorageConfig


class ConfigError(RuntimeError):
    pass


def _as_rect(data: List[int], label: str) -> Rect:
    if len(data) != 4:
        raise ConfigError(f"{label} must be [x, y, width, height], got {data}")
    return Rect(*map(int, data))


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a mapping at the top level.")
    return data


_ARUCO_DICT_NAMES = {
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_APRILTAG_36h11",
}


def _parse_geometry(data: Dict[str, Any]) -> GeometryConfig:
    override = data.get("mm_per_px_override")
    fiducial_data = data.get("fiducial")
    fiducial = None
    if fiducial_data:
        aruco_dict = fiducial_data.get("aruco_dict", "DICT_4X4_50")
        if aruco_dict not in _ARUCO_DICT_NAMES:
            raise ConfigError(f"Unsupported aruco_dict '{aruco_dict}'.")
        fiducial = FiducialConfig(
            aruco_dict=aruco_dict,
            ids=list(map(int, fiducial_data.get("ids", []))),
            tag_size_mm=float(fiducial_data["tag_size_mm"]),
        )
    return GeometryConfig(
        mm_per_px_override=float(override) if override else None,
        fiducial=fiducial,
    )


def _parse_holes(data: List[Dict[str, Any]]) -> List[HoleConfig]:
    holes = []
    for item in data:
        holes.append(
            HoleConfig(
                name=item["name"],
                roi=_as_rect(item["roi"], f"roi for hole {item['name']}"),
                nominal_diameter_mm=float(item["nominal_diameter_mm"]),
                diameter_tolerance_mm=float(item["diameter_tolerance_mm"]),
                center_tolerance_mm=float(item["center_tolerance_mm"]),
            )
        )
    return holes


def _parse_silhouette(data: Optional[Dict[str, Any]]) -> Optional[SilhouetteConfig]:
    if not data:
        return None
    return SilhouetteConfig(
        enabled=bool(data.get("enabled", True)),
        roi=_as_rect(data["roi"], "silhouette roi"),
        diff_threshold=int(data.get("diff_threshold", 25)),
        max_mismatch_pct=float(data.get("max_mismatch_pct", 2.0)),
        reference_path=Path(data["reference_path"]).resolve()
        if data.get("reference_path")
        else None,
    )


def load_config(path: Path) -> Config:
    raw = load_yaml(path)
    capture = CaptureConfig(
        fps=int(raw["capture"]["fps"]),
        resolution=(
            int(raw["capture"]["resolution"]["width"]),
            int(raw["capture"]["resolution"]["height"]),
        ),
        warmup_frames=int(raw["capture"].get("warmup_frames", 10)),
    )

    stabilization = StabilizationConfig(
        roi=_as_rect(raw["stabilization"]["roi"], "stabilization roi"),
        required_frames=int(raw["stabilization"].get("required_frames", 8)),
        pixel_jitter_threshold=float(
            raw["stabilization"].get("pixel_jitter_threshold", 2.0)
        ),
    )

    inspection = InspectionConfig(
        holes=_parse_holes(raw["inspection"].get("holes", [])),
        silhouette=_parse_silhouette(raw["inspection"].get("silhouette")),
    )

    overlay_data = raw.get("overlay", {})
    overlay = OverlayConfig(
        font_scale=float(overlay_data.get("font_scale", 0.8)),
        thickness=int(overlay_data.get("thickness", 2)),
        show_debug=bool(overlay_data.get("show_debug", False)),
    )

    storage_data = raw.get("storage", {})
    storage = StorageConfig(
        root=Path(storage_data.get("root", "data/logs")),
        save_annotated=bool(storage_data.get("save_annotated", True)),
        save_raw=bool(storage_data.get("save_raw", False)),
    )

    return Config(
        sku=str(raw["sku"]),
        camera_id=int(raw["camera_id"]),
        capture=capture,
        geometry=_parse_geometry(raw["geometry"]),
        stabilization=stabilization,
        inspection=inspection,
        overlay=overlay,
        storage=storage,
    )
