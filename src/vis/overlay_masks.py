from __future__ import annotations

from typing import Iterable, Sequence

import cv2
import numpy as np


def overlay_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color: Sequence[int] = (255, 80, 80),
    alpha: float = 0.45,
) -> np.ndarray:
    canvas = image_rgb.copy()
    binary_mask = mask.astype(bool)
    if binary_mask.ndim == 3:
        binary_mask = binary_mask.squeeze()
    color_layer = np.zeros_like(canvas)
    color_layer[binary_mask] = np.array(color, dtype=np.uint8)
    canvas = np.where(binary_mask[..., None], ((1 - alpha) * canvas + alpha * color_layer).astype(np.uint8), canvas)
    return canvas


def draw_boxes(
    image_rgb: np.ndarray,
    boxes_xyxy: Iterable[Sequence[float]],
    labels: Iterable[str] | None = None,
    color: Sequence[int] = (80, 255, 80),
) -> np.ndarray:
    canvas = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    label_list = list(labels) if labels is not None else []
    for index, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        if index < len(label_list):
            cv2.putText(
                canvas,
                label_list[index],
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

