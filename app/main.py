from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from rich import print

from app.config_loader import ConfigError, load_config
from app.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hands-free inspection with Arducam IMX296")
    parser.add_argument("--sku", required=True, help="SKU identifier (used for log output)")
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the YAML configuration for the SKU",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"[bold red]Config error:[/bold red] {exc}")
        return 1

    if cfg.sku != args.sku:
        print(
            f"[yellow]Warning:[/yellow] Config SKU '{cfg.sku}' does not match argument '{args.sku}'."
        )

    pipeline = Pipeline(cfg)
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\n[cyan]Interrupted by user[/cyan]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
