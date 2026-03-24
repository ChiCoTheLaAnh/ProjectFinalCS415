#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir, load_project_config, write_json
from src.vis.overlay_masks import draw_boxes, overlay_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate repo wiring without heavy model installs.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to the base config YAML.")
    parser.add_argument("--output-dir", default="outputs/check_env", help="Directory for synthetic artifacts.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_project_config(
        ROOT / args.config,
        ROOT / "configs" / "grounding_dino.yaml",
        ROOT / "configs" / "sam2.yaml",
    )

    output_dir = ensure_dir(ROOT / args.output_dir)

    image = np.zeros((160, 240, 3), dtype=np.uint8)
    image[..., 1] = 30
    image[40:120, 70:170, 0] = 180
    image[40:120, 70:170, 2] = 90

    mask = np.zeros((160, 240), dtype=bool)
    mask[50:115, 85:165] = True
    boxes = [[80, 45, 170, 120]]
    labels = ["synthetic-object"]

    overlaid = overlay_mask(image, mask, alpha=float(config["runtime"]["overlay_alpha"]))
    overlaid = draw_boxes(overlaid, boxes, labels)
    overlay_path = output_dir / "check_overlay.png"
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))

    summary = {
        "status": "ok",
        "config_project": config["project"]["name"],
        "active_stack": config["models"]["active_stack"],
        "artifact": str(overlay_path),
    }
    write_json(output_dir / "check_summary.json", summary)
    print(f"Environment check passed. Artifacts written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

