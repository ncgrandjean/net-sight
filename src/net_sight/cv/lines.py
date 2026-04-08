"""Line detection and classification using Hough transform."""

from __future__ import annotations

import math

import cv2
import numpy as np


def _sample_along_segment(
    img: np.ndarray, x1: int, y1: int, x2: int, y2: int, n_samples: int = 20
) -> np.ndarray:
    """Sample pixel values at evenly spaced points along a segment.

    Returns an array of shape (n_samples, C) where C is the number of
    channels (3 for BGR, 1 for grayscale).
    """
    xs = np.linspace(x1, x2, n_samples).astype(int)
    ys = np.linspace(y1, y2, n_samples).astype(int)

    h, w = img.shape[:2]
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)

    return img[ys, xs]


def _mean_color_along_segment(
    img_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int, n_samples: int = 20
) -> tuple[int, int, int]:
    """Return the mean RGB colour sampled along a line segment."""
    samples_bgr = _sample_along_segment(img_bgr, x1, y1, x2, y2, n_samples)
    mean_bgr = samples_bgr.mean(axis=0).astype(int)
    # BGR -> RGB
    return (int(mean_bgr[2]), int(mean_bgr[1]), int(mean_bgr[0]))


def detect_lines(img: np.ndarray, min_length: int = 30) -> list[dict]:
    """Detect line segments via HoughLinesP.

    Preprocessing: grayscale conversion, Canny edge detection, then
    probabilistic Hough transform.

    Returns a list of dicts with keys:
        x1, y1, x2, y2 : int endpoint coordinates
        length          : float segment length in pixels
        angle           : float angle in degrees (0-90)
        color           : (r, g, b) mean colour sampled along the segment
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min_length,
        maxLineGap=10,
    )

    if raw is None:
        return []

    # Prepare BGR image for colour sampling
    if img.ndim == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img

    lines: list[dict] = []
    for segment in raw:
        x1, y1, x2, y2 = segment[0]
        length = math.hypot(x2 - x1, y2 - y1)
        angle = math.degrees(math.atan2(abs(y2 - y1), abs(x2 - x1)))
        color = _mean_color_along_segment(img_bgr, int(x1), int(y1), int(x2), int(y2))
        lines.append({
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "length": round(length, 2),
            "angle": round(angle, 2),
            "color": color,
        })

    return lines


def classify_line_types(
    lines: list[dict], img: np.ndarray
) -> dict[str, list[dict]]:
    """Group detected lines by visual type (colour + dash pattern).

    Each line is enriched in-place with:
        color_rgb  : (r, g, b) mean colour (if not already present via ``color``)
        is_dashed  : bool

    Returns a dict mapping a human-readable type key (e.g. ``"(0,128,255)_solid"``)
    to the list of lines belonging to that group.
    """
    if img.ndim == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    for line in lines:
        # -- colour (use existing 'color' key if available) -----------------------
        if "color_rgb" not in line:
            if "color" in line:
                line["color_rgb"] = line["color"]
            else:
                line["color_rgb"] = _mean_color_along_segment(
                    img_bgr, line["x1"], line["y1"], line["x2"], line["y2"]
                )

        # -- dash detection -------------------------------------------------------
        samples_gray = _sample_along_segment(
            gray, line["x1"], line["y1"], line["x2"], line["y2"], n_samples=40
        ).astype(float)

        if samples_gray.ndim > 1:
            samples_gray = samples_gray.mean(axis=1)

        centered = samples_gray - samples_gray.mean()
        sign_changes = np.diff(np.sign(centered))
        crossings = int(np.count_nonzero(sign_changes))
        line["is_dashed"] = crossings >= 6

    # -- group by (quantised colour, dash) ------------------------------------
    groups: dict[str, list[dict]] = {}
    for line in lines:
        r, g, b = line["color_rgb"]
        # Quantise to nearest 32 to cluster similar colours together
        qr, qg, qb = (r // 32) * 32, (g // 32) * 32, (b // 32) * 32
        style = "dashed" if line.get("is_dashed") else "solid"
        key = f"({qr},{qg},{qb})_{style}"
        groups.setdefault(key, []).append(line)

    return groups


def count_connections(lines: list[dict]) -> int:
    """Return the number of distinct connections.

    Two line segments are considered part of the same connection when one
    endpoint of a segment is close (within *threshold* pixels) to an
    endpoint of another segment.  Clusters of connected segments are
    counted as single connections using a simple union-find.
    """
    if not lines:
        return 0

    n = len(lines)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    threshold = 15.0  # pixels

    # Build arrays for fast distance computation
    endpoints = np.empty((n, 4), dtype=float)
    for idx, ln in enumerate(lines):
        endpoints[idx] = [ln["x1"], ln["y1"], ln["x2"], ln["y2"]]

    for i in range(n):
        for j in range(i + 1, n):
            pts_i = [(endpoints[i, 0], endpoints[i, 1]), (endpoints[i, 2], endpoints[i, 3])]
            pts_j = [(endpoints[j, 0], endpoints[j, 1]), (endpoints[j, 2], endpoints[j, 3])]
            for pi in pts_i:
                for pj in pts_j:
                    dist = math.hypot(pi[0] - pj[0], pi[1] - pj[1])
                    if dist < threshold:
                        union(i, j)
                        break
                else:
                    continue
                break

    return len({find(i) for i in range(n)})
