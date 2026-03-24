from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from src.data.extract_frames import extract_video_frames, read_image_rgb
from src.models.grounding import load_grounding_model, predict_boxes
from src.models.sam2_wrapper import predict_image_masks, propagate_video_masks
from src.utils.io import ensure_dir, write_json
from src.vis.overlay_masks import draw_boxes, overlay_mask
from src.vis.save_video import save_video


def _merge_masks(masks: List[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    if not masks:
        return np.zeros(shape, dtype=bool)
    return np.any(np.stack([mask.astype(bool) for mask in masks], axis=0), axis=0)


def run_inference(
    input_path: str | Path,
    prompt: str,
    config: Dict[str, Any],
    output_dir: str | Path,
) -> Dict[str, Any]:
    output_root = ensure_dir(output_dir)
    runtime_cfg = config["runtime"]
    grounding_cfg = config["grounding_dino"]
    sam2_cfg = config["sam2"]

    start_time = time.time()
    suffix = Path(input_path).suffix.lower()
    model_stack = f'{config["models"]["active_stack"]["detector"]}+{config["models"]["active_stack"]["segmenter"]}'
    grounding_model = load_grounding_model(grounding_cfg)

    if suffix in {".png", ".jpg", ".jpeg", ".bmp"}:
        image_rgb = read_image_rgb(input_path)
        boxes = predict_boxes(image_rgb, prompt, grounding_cfg, model=grounding_model)
        if not boxes["boxes_xyxy"]:
            raise RuntimeError(f"No detections found for prompt: {prompt}")
        masks = predict_image_masks(image_rgb, boxes["boxes_xyxy"], sam2_cfg)
        merged_mask = _merge_masks(masks, image_rgb.shape[:2])
        overlaid = overlay_mask(image_rgb, merged_mask, alpha=float(runtime_cfg["overlay_alpha"]))
        overlaid = draw_boxes(overlaid, boxes["boxes_xyxy"], boxes["phrases"])
        image_path = output_root / "smoke_image_overlay.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))
        num_frames = 1
        artifacts = {"image_overlay": str(image_path)}
        input_type = "image"
    else:
        frames = extract_video_frames(
            input_path,
            max_frames=int(runtime_cfg["max_frames"]),
            frame_stride=int(runtime_cfg["frame_stride"]),
        )
        if not frames:
            raise RuntimeError(f"No frames could be extracted from {input_path}")
        initial_boxes = predict_boxes(frames[0], prompt, grounding_cfg, model=grounding_model)
        if not initial_boxes["boxes_xyxy"]:
            raise RuntimeError(f"No detections found in the first frame for prompt: {prompt}")
        video_masks = propagate_video_masks(frames, initial_boxes["boxes_xyxy"], sam2_cfg)
        overlaid_frames = []
        for frame_rgb, mask in zip(frames, video_masks["masks"]):
            combined = np.asarray(mask).astype(bool)
            frame_overlay = overlay_mask(frame_rgb, combined, alpha=float(runtime_cfg["overlay_alpha"]))
            frame_overlay = draw_boxes(frame_overlay, initial_boxes["boxes_xyxy"], initial_boxes["phrases"])
            overlaid_frames.append(frame_overlay)
        video_path = save_video(overlaid_frames, output_root / "smoke_video_overlay.mp4")
        num_frames = len(frames)
        artifacts = {
            "video_overlay": str(video_path),
            "video_mode": video_masks["mode"],
        }
        if "fallback_reason" in video_masks:
            artifacts["fallback_reason"] = video_masks["fallback_reason"]
        input_type = "video"

    runtime_sec = round(time.time() - start_time, 3)
    summary = {
        "prompt": prompt,
        "input_path": str(input_path),
        "input_type": input_type,
        "num_frames": num_frames,
        "runtime_sec": runtime_sec,
        "model_stack": model_stack,
        "artifacts": artifacts,
    }
    write_json(output_root / "run_summary.json", summary)
    return summary
