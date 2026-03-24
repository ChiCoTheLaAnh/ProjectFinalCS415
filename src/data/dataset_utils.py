from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_media_files(root: str | Path) -> List[Path]:
    root_path = Path(root)
    return sorted(
        path
        for path in root_path.rglob("*")
        if path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS
    )


def ensure_dataset_dirs(paths: Iterable[str | Path]) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

