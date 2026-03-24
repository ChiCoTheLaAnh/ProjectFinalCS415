#!/usr/bin/env python3
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Stub entrypoint for ablation studies.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.parse_args()
    print("run_ablation.py is scaffolded for D0 and will be implemented after the smoke test baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

