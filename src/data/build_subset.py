from __future__ import annotations

from pathlib import Path


def build_subset_manifest(source_root: str | Path, target_root: str | Path) -> dict:
    source = Path(source_root)
    target = Path(target_root)
    target.mkdir(parents=True, exist_ok=True)
    return {
        "source_root": str(source.resolve()),
        "target_root": str(target.resolve()),
        "status": "scaffold_only",
    }

