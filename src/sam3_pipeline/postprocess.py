"""Filter SAM3 masks and convert them to COCO-style bounding boxes."""

from __future__ import annotations

from typing import Any

import numpy as np


def mask_to_bbox(mask: np.ndarray) -> list[float] | None:
    """Return [x, y, width, height] around non-zero pixels."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum(dtype=np.int64)
    union = np.logical_or(first, second).sum(dtype=np.int64)
    return float(intersection / union) if union else 0.0


def _valid_region_mask(
    height: int, width: int, normalized_region: list[float]
) -> np.ndarray:
    if len(normalized_region) != 4:
        raise ValueError("valid_region must contain [x, y, width, height].")
    x, y, region_width, region_height = (float(value) for value in normalized_region)
    if (
        x < 0
        or y < 0
        or region_width <= 0
        or region_height <= 0
        or x + region_width > 1
        or y + region_height > 1
    ):
        raise ValueError("valid_region must be a normalized rectangle inside the image.")
    x1, y1 = round(x * width), round(y * height)
    x2, y2 = round((x + region_width) * width), round((y + region_height) * height)
    valid = np.zeros((height, width), dtype=bool)
    valid[y1:y2, x1:x2] = True
    return valid


def filter_proposals(
    proposals: list[dict[str, Any]],
    *,
    image_height: int,
    image_width: int,
    score_threshold: float,
    min_mask_area: int,
    duplicate_iou_threshold: float,
    valid_region: list[float],
    valid_region_min_fraction: float,
) -> list[dict[str, Any]]:
    """Apply deterministic filters, keeping the highest scoring duplicate."""
    region = _valid_region_mask(image_height, image_width, valid_region)
    candidates: list[dict[str, Any]] = []

    for proposal in proposals:
        mask = np.asarray(proposal["mask"], dtype=bool)
        if mask.shape != (image_height, image_width):
            raise ValueError(
                f"Mask shape {mask.shape} does not match image "
                f"{(image_height, image_width)}."
            )
        score = float(proposal["score"])
        area = int(mask.sum(dtype=np.int64))
        inside = int(np.logical_and(mask, region).sum(dtype=np.int64))
        inside_fraction = inside / area if area else 0.0
        bbox = mask_to_bbox(mask)
        if (
            score < score_threshold
            or area < min_mask_area
            or bbox is None
            or inside_fraction < valid_region_min_fraction
        ):
            continue
        normalized = dict(proposal)
        normalized.update(
            {
                "mask": mask,
                "score": score,
                "area": area,
                "valid_region_fraction": inside_fraction,
                "bbox": bbox,
            }
        )
        candidates.append(normalized)

    kept: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        if any(
            mask_iou(candidate["mask"], other["mask"]) >= duplicate_iou_threshold
            for other in kept
        ):
            continue
        kept.append(candidate)
    return kept
