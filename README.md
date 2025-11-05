# Arducam IMX296 Inspection MVP

Hands-free GREEN/RED inspection app for the Arducam IMX296 USB camera. Provides a full-screen operator UI, automatic capture once the fixture is stable, dimensional checks for hole ROIs, optional silhouette diff, and persistent run logs.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python app/main.py --sku tf-1234 --config config/sku_tf-1234.yaml
```

## Adding a new part

Launch the helper and follow the prompts to create a fresh YAML:

```bash
python tools/add_part.py
```

When you're done, point the inspection app at the generated config. You can also click the `[+]` button in the inspection window to launch the same helper without leaving the operator UI.

## Configuration

Each SKU uses a YAML file under `config/`. Set the `camera_id` to match your system, tweak ROI boxes, tolerances, and optional silhouette diff thresholds.  See `config/sku_tf-1234.yaml` for a ready-made example.

## Outputs

Annotated PNGs and JSON reports are stored in `data/logs/<sku>/<timestamp>/` for traceability.

## Hardware setup

* Arducam IMX296 USB 3.0
* Fixture with fiducials (ArUco IDs 10 and 11, 20 mm squares) for auto-scale or set a manual `geometry.mm_per_px_override`

## Developing

* Core app lives under `app/`
* Entry point: `app/main.py`
* Logging to CLI + rich window overlay
