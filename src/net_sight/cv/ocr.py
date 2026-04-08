"""OCR extraction using EasyOCR with lazy reader initialisation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import easyocr

# Module-level singleton: initialised once on first call to get_reader().
_reader: easyocr.Reader | None = None


def get_reader(langs: list[str] | None = None) -> easyocr.Reader:
    """Return a shared EasyOCR reader, creating it on first call.

    EasyOCR model loading is expensive, so the reader is cached at module
    level.
    """
    global _reader  # noqa: PLW0603
    if _reader is None:
        import easyocr

        _reader = easyocr.Reader(langs or ["en"], gpu=False)
    return _reader


def extract_text(img: np.ndarray) -> list[dict]:
    """Extract all visible texts from *img* using EasyOCR.

    Returns a list of dicts:
        text       : recognised string
        x, y       : int top-left corner of the bounding box
        width      : int bounding box width
        height     : int bounding box height
        confidence : float 0-1
    """
    reader = get_reader()
    results = reader.readtext(img)

    texts: list[dict] = []
    for bbox_pts, text, confidence in results:
        xs = [int(p[0]) for p in bbox_pts]
        ys = [int(p[1]) for p in bbox_pts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        texts.append({
            "text": text,
            "x": x_min,
            "y": y_min,
            "width": x_max - x_min,
            "height": y_max - y_min,
            "confidence": round(float(confidence), 4),
        })

    return texts


# Keep the old name as an alias for backward compatibility.
extract_texts = extract_text


def extract_text_in_region(
    img: np.ndarray, x: int, y: int, w: int, h: int
) -> list[dict]:
    """Extract texts within a rectangular region of *img*.

    Crops the image to the region ``(x, y, w, h)`` and runs OCR on the
    crop.  Returned coordinates are remapped to the full image frame.
    """
    img_h, img_w = img.shape[:2]
    # Clamp to image bounds
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(img_w, x + w)
    y1 = min(img_h, y + h)

    if x1 <= x0 or y1 <= y0:
        return []

    crop = img[y0:y1, x0:x1]
    texts = extract_text(crop)

    # Remap to global coordinates
    for t in texts:
        t["x"] += x0
        t["y"] += y0

    return texts


def extract_texts_from_tile(tile_img: np.ndarray, tile_meta: dict) -> list[dict]:
    """Like :func:`extract_text` but remaps coordinates to the global image frame.

    *tile_meta* must contain ``"x"`` and ``"y"`` keys representing the
    top-left offset of the tile inside the full image.
    """
    texts = extract_text(tile_img)
    ox, oy = int(tile_meta["x"]), int(tile_meta["y"])
    for t in texts:
        t["x"] += ox
        t["y"] += oy
    return texts


def group_text_by_proximity(
    texts: list[dict], max_gap: float = 30.0
) -> list[list[dict]]:
    """Group text detections that are spatially close together.

    Two texts are considered neighbours when the gap between their
    bounding boxes is smaller than *max_gap* pixels.  Connected groups
    (multi-line labels, compound names) are returned together.

    Uses a simple union-find to cluster.
    """
    if not texts:
        return []

    n = len(texts)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if _bbox_gap(texts[i], texts[j]) < max_gap:
                union(i, j)

    groups: dict[int, list[dict]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(texts[i])

    # Sort each group top-to-bottom, left-to-right
    result: list[list[dict]] = []
    for group in groups.values():
        group.sort(key=lambda t: (t["y"], t["x"]))
        result.append(group)

    # Sort groups by the position of their first element
    result.sort(key=lambda g: (g[0]["y"], g[0]["x"]))
    return result


def _bbox_gap(a: dict, b: dict) -> float:
    """Compute the minimum gap between two axis-aligned bounding boxes.

    Each dict has ``x, y, width, height``.  Returns 0.0 when boxes
    overlap.
    """
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["width"], ay1 + a["height"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["width"], by1 + b["height"]

    dx = max(0, max(ax1 - bx2, bx1 - ax2))
    dy = max(0, max(ay1 - by2, by1 - ay2))

    return math.hypot(dx, dy)
