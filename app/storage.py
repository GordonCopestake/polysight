from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import cv2

from .config_loader import StorageConfig
from .inspection import InspectionResult
from .utils import ensure_dir, timestamp


class Storage:
    def __init__(self, cfg: StorageConfig, sku: str):
        self._cfg = cfg
        self._sku = sku
        self._run_dir: Path | None = None

    def _ensure_run_dir(self) -> Path:
        if self._run_dir is None:
            base = ensure_dir(self._cfg.root / self._sku)
            self._run_dir = ensure_dir(base / timestamp())
        return self._run_dir

    def reset_run(self) -> None:
        self._run_dir = None

    def save(self, result: InspectionResult, frame_bgr: Any) -> Path:
        run_dir = self._ensure_run_dir()
        if self._cfg.save_raw:
            cv2.imwrite(str(run_dir / "raw.png"), frame_bgr)
        if self._cfg.save_annotated:
            cv2.imwrite(str(run_dir / "annotated.png"), result.annotated)
        self._save_json(run_dir / "result.json", result)
        return run_dir

    def _save_json(self, path: Path, result: InspectionResult) -> None:
        payload: Dict[str, Any] = {
            "passed": result.passed,
            "reasons": result.reasons,
            "holes": [
                {
                    "name": hole.name,
                    "diameter_mm": hole.diameter_mm,
                    "diameter_error_mm": hole.diameter_error_mm,
                    "center_offset_mm": hole.center_offset_mm,
                    "passed": hole.passed,
                    "reason": hole.reason,
                    "roi": asdict(hole.roi),
                }
                for hole in result.holes
            ],
        }
        if result.silhouette:
            payload["silhouette"] = {
                "passed": result.silhouette.passed,
                "mismatch_pct": result.silhouette.mismatch_pct,
                "reason": result.silhouette.reason,
                "roi": asdict(result.silhouette.roi),
            }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
