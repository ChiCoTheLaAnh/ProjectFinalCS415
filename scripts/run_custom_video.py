#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.pipeline import run_inference
from src.utils.io import load_project_config
from src.utils.logger import configure_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run image or short-video smoke inference.")
    parser.add_argument("--config", default="configs/base.yaml", help="Base runtime config.")
    parser.add_argument("--input_video", required=True, help="Path to the input image or video.")
    parser.add_argument("--prompt", required=True, help="Text prompt for GroundingDINO.")
    parser.add_argument("--output_dir", required=True, help="Directory for saved artifacts.")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional max frame override for videos.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = configure_logger()
    config = load_project_config(
        ROOT / args.config,
        ROOT / "configs" / "grounding_dino.yaml",
        ROOT / "configs" / "sam2.yaml",
    )
    if args.max_frames is not None:
        config["runtime"]["max_frames"] = args.max_frames

    summary = run_inference(
        input_path=args.input_video,
        prompt=args.prompt,
        config=config,
        output_dir=args.output_dir,
    )
    logger.info("Smoke run complete: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

