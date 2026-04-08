"""Auto-tuning of tiling parameters based on image characteristics.

Uses analysis results from :mod:`net_sight.preprocess.autotune` to decide how
many tiles, how much overlap, and whether smart (zone-based) tiling should be
preferred over a regular grid.
"""

from __future__ import annotations

import math

import numpy as np

from net_sight.tiling.smart import detect_zones


def compute_tiling_params(img: np.ndarray, characteristics: dict) -> dict:
    """Derive optimal tiling parameters from image characteristics.

    Args:
        img: Source image (used to probe for logical zones).
        characteristics: Dict produced by
            :func:`net_sight.preprocess.autotune.analyze_image`, expected keys:

            * ``line_density`` (float): ratio of foreground pixels.
            * ``text_density`` (float): ratio of text-like pixels.
            * ``mean_line_thickness`` (float): average line width in px.
            * ``num_distinct_colors`` (int): number of colour clusters found.

    Returns:
        Dict with keys:

        * ``rows`` (int): number of grid rows.
        * ``cols`` (int): number of grid columns.
        * ``overlap`` (float): overlap fraction (0-1).
        * ``use_smart_tiling`` (bool): True when zone-based tiling is recommended.
    """
    line_density: float = characteristics.get("line_density", 0.05)
    text_density: float = characteristics.get("text_density", 0.02)
    mean_thickness: float = characteristics.get("mean_line_thickness", 2.0)
    num_colors: int = characteristics.get("num_distinct_colors", 1)

    # --- Rows / cols ----------------------------------------------------------
    # Denser diagrams need more tiles to preserve detail.
    combined_density = line_density + text_density

    if combined_density > 0.20:
        target_tiles = 16
    elif combined_density > 0.10:
        target_tiles = 12
    elif combined_density > 0.04:
        target_tiles = 8
    else:
        target_tiles = 4

    # Use image aspect ratio to distribute tiles between rows and cols.
    img_h, img_w = img.shape[:2]
    aspect = img_w / max(img_h, 1)

    # Start from a square-ish grid, then skew towards the dominant axis.
    base = math.sqrt(target_tiles)
    if aspect >= 1.0:
        cols = max(1, round(base * math.sqrt(aspect)))
        rows = max(1, round(target_tiles / cols))
    else:
        rows = max(1, round(base / math.sqrt(aspect)))
        cols = max(1, round(target_tiles / rows))

    # Ensure at least 2x2 if target_tiles >= 4.
    if target_tiles >= 4:
        rows = max(2, rows)
        cols = max(2, cols)

    # --- Overlap --------------------------------------------------------------
    # More overlap when: high text density (long labels may straddle tiles),
    # thin lines (easily cut), or many colours (complex diagram).
    base_overlap = 0.20

    if text_density > 0.05:
        base_overlap += 0.05
    if mean_thickness < 2.0:
        base_overlap += 0.05
    if num_colors > 6:
        base_overlap += 0.05

    overlap = min(base_overlap, 0.40)  # Cap at 40% to avoid excessive redundancy.

    # --- Smart tiling probe ---------------------------------------------------
    # Run a quick zone detection. If enough distinct zones are found, recommend
    # zone-based tiling instead of a plain grid.
    use_smart_tiling = False
    try:
        zones = detect_zones(img)
        # Require at least 3 zones that together cover a meaningful portion
        # of the image, and fewer zones than the planned grid tiles (otherwise
        # the grid already captures the detail).
        if len(zones) >= 3:
            total_zone_area = sum(w * h for _, _, w, h in zones)
            img_area = img_h * img_w
            coverage = total_zone_area / max(img_area, 1)
            if 0.30 <= coverage <= 1.5 and len(zones) <= rows * cols:
                use_smart_tiling = True
    except Exception:
        # Zone detection is best-effort; fall back to grid.
        pass

    return {
        "rows": rows,
        "cols": cols,
        "overlap": round(overlap, 2),
        "use_smart_tiling": use_smart_tiling,
    }
