from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from src.utils.prompts import normalize_prompt


def _resolve_device(device_name: str) -> str:
    if device_name != "auto":
        return device_name
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_grounding_model(config: Dict[str, Any]):
    config_path = Path(config["config_path"])
    checkpoint_path = Path(config["checkpoint_path"])
    if not config_path.exists():
        raise FileNotFoundError(f"GroundingDINO config file not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"GroundingDINO checkpoint not found: {checkpoint_path}. Run `bash setup_colab.sh --with-models` first."
        )

    try:
        from groundingdino.util.inference import load_model
    except ImportError as exc:
        raise RuntimeError(
            "GroundingDINO is not installed. Run `bash setup_colab.sh --with-models` in Colab first."
        ) from exc

    device = _resolve_device(config.get("device", "auto"))
    signature = inspect.signature(load_model)
    kwargs = {"device": device} if "device" in signature.parameters else {}
    model = load_model(str(config_path), str(checkpoint_path), **kwargs)
    if hasattr(model, "to"):
        model = model.to(device)
    return model


def _normalized_cxcywh_to_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0, 4), dtype=float)
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - (w / 2.0)) * width
    y1 = (cy - (h / 2.0)) * height
    x2 = (cx + (w / 2.0)) * width
    y2 = (cy + (h / 2.0)) * height
    xyxy = np.stack([x1, y1, x2, y2], axis=1)
    xyxy[:, 0::2] = np.clip(xyxy[:, 0::2], 0, width - 1)
    xyxy[:, 1::2] = np.clip(xyxy[:, 1::2], 0, height - 1)
    return xyxy


def predict_boxes(
    image_rgb: np.ndarray,
    prompt: str,
    config: Dict[str, Any],
    model=None,
) -> Dict[str, List[Any]]:
    if model is None:
        model = load_grounding_model(config)

    try:
        from groundingdino.util.inference import load_image, predict
    except ImportError as exc:
        raise RuntimeError(
            "GroundingDINO inference helpers are unavailable. Confirm the official repo install succeeded."
        ) from exc

    normalized_prompt = normalize_prompt(prompt)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "frame.png"
        cv2.imwrite(str(temp_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        _, image_tensor = load_image(str(temp_path))

    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=normalized_prompt,
        box_threshold=float(config["box_threshold"]),
        text_threshold=float(config["text_threshold"]),
    )

    if hasattr(boxes, "cpu"):
        boxes = boxes.cpu().numpy()
    if hasattr(logits, "cpu"):
        logits = logits.cpu().numpy()

    height, width = image_rgb.shape[:2]
    boxes_array = np.asarray(boxes)
    if boxes_array.size == 0:
        return {
            "boxes_xyxy": [],
            "scores": [],
            "phrases": [],
            "prompt": normalized_prompt,
        }

    boxes_xyxy = _normalized_cxcywh_to_xyxy(boxes_array, width=width, height=height)
    return {
        "boxes_xyxy": boxes_xyxy.tolist(),
        "scores": np.asarray(logits).astype(float).tolist(),
        "phrases": list(phrases),
        "prompt": normalized_prompt,
    }
