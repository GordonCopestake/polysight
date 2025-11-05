from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from rich.console import Console
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt


console = Console()
CONFIG_DIR = Path("config")
DEFAULT_CAPTURE = {
    "fps": 30,
    "resolution": {"width": 1440, "height": 1080},
    "warmup_frames": 15,
}
DEFAULT_STABILIZATION = {
    "roi": [480, 320, 480, 320],
    "required_frames": 8,
    "pixel_jitter_threshold": 2.5,
}

SUPPORTED_ARUCO = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_APRILTAG_36h11",
]


@dataclass
class HoleBlueprint:
    name: str
    roi: List[int]
    nominal_diameter_mm: float
    diameter_tolerance_mm: float
    center_tolerance_mm: float


@dataclass
class SilhouetteBlueprint:
    roi: List[int]
    diff_threshold: int
    max_mismatch_pct: float
    reference_path: Optional[str]


def prompt_sku() -> str:
    while True:
        sku = Prompt.ask("Item ID (e.g. 1603B-00)").strip()
        if sku:
            return sku
        console.print("[red]SKU cannot be empty[/red]")


def prompt_output_path(sku: str) -> Path:
    default_name = f"sku_{sku.lower().replace(' ', '_').replace('-', '_')}" + ".yaml"
    default_path = CONFIG_DIR / default_name
    answer = Prompt.ask("Output YAML", default=str(default_path))
    path = Path(answer)
    if not path.suffix:
        path = path.with_suffix(".yaml")
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def prompt_capture() -> Dict:
    console.print("\n[bold]Camera Capture[/bold]")
    camera_id = IntPrompt.ask("Camera ID", default=0)
    fps = IntPrompt.ask("FPS", default=DEFAULT_CAPTURE["fps"])
    width = IntPrompt.ask("Resolution width", default=DEFAULT_CAPTURE["resolution"]["width"])
    height = IntPrompt.ask("Resolution height", default=DEFAULT_CAPTURE["resolution"]["height"])
    warmup = IntPrompt.ask("Warmup frames", default=DEFAULT_CAPTURE["warmup_frames"])
    return {
        "camera_id": camera_id,
        "capture": {"fps": fps, "resolution": {"width": width, "height": height}, "warmup_frames": warmup},
    }


def prompt_geometry() -> Dict:
    console.print("\n[bold]Geometry Scaling[/bold]")
    use_override = Confirm.ask("Do you know mm per pixel already?", default=False)
    override = None
    fiducial = None
    if use_override:
        override = FloatPrompt.ask("mm per pixel", default=0.05)
    else:
        if Confirm.ask("Use ArUco fiducials for scaling?", default=True):
            dict_name = Prompt.ask(
                "ArUco dictionary",
                choices=SUPPORTED_ARUCO,
                default=SUPPORTED_ARUCO[0],
                show_choices=False,
            )
            tag_ids_raw = Prompt.ask("Marker IDs (comma separated)", default="10,11")
            tag_ids = [int(part.strip()) for part in tag_ids_raw.split(",") if part.strip()]
            tag_size = FloatPrompt.ask("Marker size (mm)", default=20.0)
            fiducial = {
                "aruco_dict": dict_name,
                "ids": tag_ids,
                "tag_size_mm": tag_size,
            }
    return {"mm_per_px_override": override, "fiducial": fiducial}


def prompt_stabilization() -> Dict:
    console.print("\n[bold]Stabilization ROI[/bold]")
    roi_default = DEFAULT_STABILIZATION["roi"]
    use_default = Confirm.ask(
        f"Use default ROI {roi_default}?", default=True
    )
    if use_default:
        roi = roi_default
    else:
        roi = [
            IntPrompt.ask("ROI X", default=roi_default[0]),
            IntPrompt.ask("ROI Y", default=roi_default[1]),
            IntPrompt.ask("ROI width", default=roi_default[2]),
            IntPrompt.ask("ROI height", default=roi_default[3]),
        ]
    required_frames = IntPrompt.ask("Frames to hold steady", default=DEFAULT_STABILIZATION["required_frames"])
    jitter = FloatPrompt.ask("Pixel jitter threshold", default=DEFAULT_STABILIZATION["pixel_jitter_threshold"])
    return {"roi": roi, "required_frames": required_frames, "pixel_jitter_threshold": jitter}


def prompt_holes() -> List[HoleBlueprint]:
    console.print("\n[bold]Hole Features[/bold]")
    holes: List[HoleBlueprint] = []
    while Confirm.ask("Add a hole?", default=len(holes) == 0):
        name = Prompt.ask("Hole name", default=f"hole_{len(holes)+1}")
        roi = [
            IntPrompt.ask("ROI X"),
            IntPrompt.ask("ROI Y"),
            IntPrompt.ask("ROI width"),
            IntPrompt.ask("ROI height"),
        ]
        nominal = FloatPrompt.ask("Nominal diameter (mm)")
        tol_d = FloatPrompt.ask("Diameter tolerance (mm)", default=0.25)
        tol_c = FloatPrompt.ask("Center tolerance (mm)", default=0.35)
        holes.append(
            HoleBlueprint(
                name=name,
                roi=roi,
                nominal_diameter_mm=nominal,
                diameter_tolerance_mm=tol_d,
                center_tolerance_mm=tol_c,
            )
        )
    return holes


def prompt_silhouette() -> Optional[SilhouetteBlueprint]:
    console.print("\n[bold]Silhouette Diff[/bold]")
    if not Confirm.ask("Configure silhouette diff?", default=False):
        return None
    roi = [
        IntPrompt.ask("ROI X"),
        IntPrompt.ask("ROI Y"),
        IntPrompt.ask("ROI width"),
        IntPrompt.ask("ROI height"),
    ]
    threshold = IntPrompt.ask("Difference threshold", default=25)
    mismatch = FloatPrompt.ask("Max mismatch percent", default=2.0)
    reference = Prompt.ask("Reference mask path (optional)", default="")
    ref_path = reference.strip() or None
    return SilhouetteBlueprint(
        roi=roi,
        diff_threshold=threshold,
        max_mismatch_pct=mismatch,
        reference_path=ref_path,
    )


def build_config(
    sku: str,
    capture: Dict,
    geometry: Dict,
    stabilization: Dict,
    holes: List[HoleBlueprint],
    silhouette: Optional[SilhouetteBlueprint],
) -> Dict:
    inspection: Dict[str, object] = {
        "holes": [hole.__dict__ for hole in holes],
    }
    if silhouette:
        inspection["silhouette"] = {
            "enabled": True,
            "roi": silhouette.roi,
            "diff_threshold": silhouette.diff_threshold,
            "max_mismatch_pct": silhouette.max_mismatch_pct,
            "reference_path": silhouette.reference_path,
        }
    else:
        inspection["silhouette"] = {
            "enabled": False,
            "roi": [0, 0, 0, 0],
            "diff_threshold": 25,
            "max_mismatch_pct": 2.0,
            "reference_path": None,
        }

    return {
        "sku": sku,
        "camera_id": capture["camera_id"],
        "capture": capture["capture"],
        "geometry": geometry,
        "stabilization": stabilization,
        "inspection": inspection,
        "overlay": {"font_scale": 0.8, "thickness": 2, "show_debug": True},
        "storage": {"root": "data/logs", "save_annotated": True, "save_raw": False},
    }


def save_config(path: Path, config: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)


def main() -> int:
    console.print("[bold cyan]Add a new inspection part[/bold cyan]")
    sku = prompt_sku()
    output = prompt_output_path(sku)
    capture = prompt_capture()
    geometry = prompt_geometry()
    stabilization = prompt_stabilization()
    holes = prompt_holes()
    silhouette = prompt_silhouette()

    if not holes:
        if not Confirm.ask("No holes defined. Continue?", default=True):
            console.print("[yellow]Aborted[/yellow]")
            return 1

    config = build_config(sku, capture, geometry, stabilization, holes, silhouette)
    save_config(output, config)
    console.print(f"[green]Config saved to[/green] {output}")
    console.print("You can launch the inspection with:")
    console.print(f"  python app/main.py --sku {sku} --config {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
