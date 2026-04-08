"""Smart zone-based tiling using contour detection.

Instead of a uniform grid, this module detects logical zones in the diagram
(coloured frames, bordered sections) and creates one tile per zone. Useful
when the network diagram has clearly separated sub-areas that do not align
well with a regular grid.
"""

from __future__ import annotations

import cv2
import numpy as np

from net_sight.tiling.grid import Tile


def detect_zones(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect rectangular zones in *img* using contour analysis.

    The pipeline finds large closed contours (coloured frames, bordered
    sections) and returns their bounding rectangles.

    Args:
        img: Source image (BGR or grayscale).

    Returns:
        List of (x, y, w, h) bounding boxes, sorted top-to-bottom then
        left-to-right, with small/duplicate regions filtered out.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    img_h, img_w = gray.shape[:2]
    img_area = img_h * img_w

    # Adaptive threshold handles varying background colours better than a
    # global threshold for diagrams with multiple coloured zones.
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
    )

    # Morphological close to connect nearby edges into solid frames.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Minimum zone area: 1% of image; maximum: 95% (skip near-full-image frames).
    min_area = img_area * 0.01
    max_area = img_area * 0.95

    raw_boxes: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area or area > max_area:
            continue
        # Reject very elongated slices (aspect ratio > 8:1 either way).
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > 8.0:
            continue
        raw_boxes.append((int(x), int(y), int(w), int(h)))

    # Merge overlapping / nearly-identical boxes (IoU > 0.5).
    zones = _merge_overlapping(raw_boxes, iou_threshold=0.5)

    # Sort top-to-bottom, left-to-right.
    zones.sort(key=lambda b: (b[1], b[0]))
    return zones


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Compute Intersection-over-Union for two (x, y, w, h) boxes."""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _merge_overlapping(
    boxes: list[tuple[int, int, int, int]],
    iou_threshold: float,
) -> list[tuple[int, int, int, int]]:
    """Greedily merge boxes whose IoU exceeds *iou_threshold*.

    When two boxes overlap, they are replaced by the bounding box that
    contains both. The process repeats until no more merges occur.
    """
    if not boxes:
        return []

    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        new_merged: list[tuple[int, int, int, int]] = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            current = merged[i]
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                if _iou(current, merged[j]) > iou_threshold:
                    # Merge into bounding box of both.
                    x1 = min(current[0], merged[j][0])
                    y1 = min(current[1], merged[j][1])
                    x2 = max(current[0] + current[2], merged[j][0] + merged[j][2])
                    y2 = max(current[1] + current[3], merged[j][1] + merged[j][3])
                    current = (x1, y1, x2 - x1, y2 - y1)
                    used[j] = True
                    changed = True
            new_merged.append(current)
            used[i] = True
        merged = new_merged
    return merged


def split_by_zones(
    img: np.ndarray,
    zones: list[tuple[int, int, int, int]],
    target_size: int = 1024,
) -> list[Tile]:
    """Create one tile per detected zone, resizing if necessary.

    Each tile is cropped from *img* at the zone coordinates. If a zone
    dimension exceeds *target_size*, the tile image is downscaled so its
    longest side equals *target_size* (the positional metadata still refers to
    the original image coordinates).

    Args:
        img: Source image (H x W x C or H x W).
        zones: List of (x, y, w, h) bounding boxes from :func:`detect_zones`.
        target_size: Maximum pixel size for the longest side of a tile image.

    Returns:
        List of :class:`Tile` objects, one per zone, ordered as *zones*.
    """
    img_h, img_w = img.shape[:2]
    tiles: list[Tile] = []

    for idx, (x, y, w, h) in enumerate(zones):
        # Clamp to image bounds.
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        if w <= 0 or h <= 0:
            continue

        crop = img[y : y + h, x : x + w].copy()

        # Downscale if the crop is larger than target_size.
        longest = max(w, h)
        if longest > target_size:
            scale = target_size / longest
            new_w = max(1, round(w * scale))
            new_h = max(1, round(h * scale))
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        tiles.append(
            Tile(
                image=crop,
                row=idx,
                col=0,
                x=x,
                y=y,
                width=w,
                height=h,
            )
        )

    return tiles
