from __future__ import annotations

from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np


def save_video(frames: Iterable[np.ndarray], output_path: str | Path, fps: int = 12) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(target, list(frames), fps=fps, macro_block_size=None)
    return target

