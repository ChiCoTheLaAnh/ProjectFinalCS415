from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np


def read_image_rgb(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def extract_video_frames(
    video_path: str | Path,
    max_frames: int | None = None,
    frame_stride: int = 1,
) -> List[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    frames: List[np.ndarray] = []
    frame_index = 0
    while True:
        success, frame_bgr = capture.read()
        if not success:
            break
        if frame_index % frame_stride == 0:
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        frame_index += 1
        if max_frames is not None and len(frames) >= max_frames:
            break
    capture.release()
    return frames


def save_frames(frames: List[np.ndarray], output_dir: str | Path, prefix: str = "frame") -> List[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for index, frame_rgb in enumerate(frames):
        frame_path = output_path / f"{prefix}_{index:04d}.png"
        cv2.imwrite(str(frame_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        written.append(frame_path)
    return written

