"""Shape detection via contour analysis."""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Low-level contour helpers
# ---------------------------------------------------------------------------

def _classify_contour(contour: np.ndarray) -> str:
    """Classify a contour as rectangle, circle, or polygon."""
    perimeter = cv2.arcLength(contour, closed=True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, closed=True)
    vertices = len(approx)

    if vertices == 4:
        return "rectangle"

    if perimeter > 0:
        area = cv2.contourArea(contour)
        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        if circularity > 0.75:
            return "circle"

    return "polygon"


def _contour_to_dict(contour: np.ndarray, shape_type: str | None = None) -> dict:
    """Convert a single contour into a serialisable dict.

    Keys: type, x, y, width, height, area, center.
    """
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        cx, cy = x + w // 2, y + h // 2
    else:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

    stype = shape_type or _classify_contour(contour)

    return {
        "type": stype,
        "x": int(x),
        "y": int(y),
        "width": int(w),
        "height": int(h),
        "area": round(float(area), 2),
        "center": [int(cx), int(cy)],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_shapes(img: np.ndarray, min_area: int = 500) -> list[dict]:
    """Detect and classify shapes (rectangle, circle, polygon).

    Pipeline: grayscale -> blur -> Canny -> find contours -> classify.

    Returns a list of dicts:
        type   : "rectangle" | "circle" | "polygon"
        bbox   : [x, y, w, h]  (kept for backward compatibility)
        area   : contour area in pixels
        center : [cx, cy]
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes: list[dict] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        d = _contour_to_dict(cnt)
        # Legacy "bbox" key for backward compat with detect_shapes callers
        d["bbox"] = [d["x"], d["y"], d["width"], d["height"]]
        shapes.append(d)

    return shapes


def detect_rectangles(img: np.ndarray, min_area: int = 500) -> list[dict]:
    """Detect rectangular contours in *img*.

    Returns a list of dicts with keys:
        x, y, width, height, area
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects: list[dict] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(cnt, closed=True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, closed=True)
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        rects.append({
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "area": round(float(area), 2),
        })

    return rects


def detect_icons(
    img: np.ndarray,
    min_area: int = 100,
    max_area: int = 5000,
) -> list[dict]:
    """Detect small distinct shapes likely to be equipment icons.

    Filters contours by area (between *min_area* and *max_area*) to
    isolate icon-sized elements.

    Returns a list of dicts:
        x, y, width, height, area, center
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    icons: list[dict] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        moments = cv2.moments(cnt)
        if moments["m00"] == 0:
            cx, cy = x + w // 2, y + h // 2
        else:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])

        icons.append({
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "area": round(float(area), 2),
            "center": [int(cx), int(cy)],
        })

    return icons


def classify_shapes(rectangles: list[dict]) -> dict[str, list[dict]]:
    """Classify rectangles into "zones" (large) and "devices" (small).

    The threshold is adaptive: rectangles whose area is above the median
    are considered zones, those below are devices.  When there are fewer
    than 3 rectangles the 75th-percentile of image area heuristic is
    skipped in favour of an absolute area threshold of 50 000 px.
    """
    if not rectangles:
        return {"zones": [], "devices": []}

    areas = [r["area"] for r in rectangles]

    if len(areas) >= 3:
        threshold = float(np.median(areas))
    else:
        threshold = 50_000.0

    zones: list[dict] = []
    devices: list[dict] = []
    for rect in rectangles:
        if rect["area"] >= threshold:
            zones.append(rect)
        else:
            devices.append(rect)

    return {"zones": zones, "devices": devices}
