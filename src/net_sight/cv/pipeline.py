"""CV augmentation pipeline: orchestrates all computer-vision analyses."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from net_sight.cv.colors import cluster_colors, extract_line_colors, build_color_legend
from net_sight.cv.lines import classify_line_types, count_connections, detect_lines
from net_sight.cv.ocr import extract_text, group_text_by_proximity
from net_sight.cv.shapes import classify_shapes, detect_icons, detect_rectangles


@dataclass
class CVResults:
    """Container for all CV analysis outputs.

    Every field stores plain Python types (no numpy objects) so that the
    whole structure is directly JSON-serialisable.
    """

    lines: list[dict] = field(default_factory=list)
    texts: list[dict] = field(default_factory=list)
    shapes: dict = field(default_factory=dict)
    color_clusters: dict[str, tuple[int, int, int]] = field(default_factory=dict)

    # Derived / summary fields populated by the pipeline
    line_groups: dict[str, list[dict]] = field(default_factory=dict)
    text_groups: list[list[dict]] = field(default_factory=list)
    icons: list[dict] = field(default_factory=list)
    connection_count: int = 0
    color_legend: dict[str, str] = field(default_factory=dict)


def run_cv_analysis(img: np.ndarray) -> CVResults:
    """Execute the full CV augmentation pipeline on *img*.

    Steps executed:
    1. Line detection + classification + connection counting
    2. OCR text extraction + proximity grouping
    3. Rectangle and icon detection + shape classification
    4. Colour clustering along detected lines + legend building
    """
    results = CVResults()

    # 1. Lines
    results.lines = detect_lines(img)
    results.line_groups = classify_line_types(results.lines, img)
    results.connection_count = count_connections(results.lines)

    # 2. OCR
    results.texts = extract_text(img)
    results.text_groups = group_text_by_proximity(results.texts)

    # 3. Shapes
    rectangles = detect_rectangles(img)
    results.icons = detect_icons(img)
    results.shapes = classify_shapes(rectangles)

    # 4. Colours
    if results.lines:
        colors = extract_line_colors(img, results.lines)
        results.color_clusters = cluster_colors(colors)
        results.color_legend = build_color_legend(results.color_clusters)

    return results


def run_cv_on_tile(tile_img: np.ndarray) -> CVResults:
    """Run the CV pipeline on a single tile image.

    Identical to :func:`run_cv_analysis` but named explicitly for
    tile-level usage.  Callers are responsible for coordinate remapping
    if needed.
    """
    return run_cv_analysis(tile_img)


def format_cv_context(results: CVResults) -> str:
    """Format CV results as a human-readable text block for VLM prompt injection.

    The output is designed to be appended to a VLM prompt so the model
    can leverage structured data extracted via classical CV.
    """
    parts: list[str] = []

    # -- Lines ----------------------------------------------------------------
    n_lines = len(results.lines)
    n_groups = len(results.line_groups)
    parts.append(
        f"{n_lines} line segments detected, "
        f"grouped into {n_groups} visual categories."
    )
    parts.append(f"Estimated distinct connections: {results.connection_count}.")

    if results.line_groups:
        parts.append("Line categories:")
        for key, members in results.line_groups.items():
            parts.append(f"  - {key}: {len(members)} segments")

    # -- Colours --------------------------------------------------------------
    if results.color_legend:
        parts.append("Colour legend (auto-detected):")
        for cid, label in results.color_legend.items():
            rgb = results.color_clusters.get(cid, (0, 0, 0))
            parts.append(f"  - {label}: RGB{rgb}")

    # -- OCR ------------------------------------------------------------------
    n_texts = len(results.texts)
    parts.append(f"OCR detected {n_texts} text elements.")

    if results.texts:
        # List up to 30 texts to keep the prompt manageable
        sample = results.texts[:30]
        labels = [t["text"] for t in sample]
        parts.append("Visible labels: " + ", ".join(f'"{l}"' for l in labels))
        if n_texts > 30:
            parts.append(f"  ... and {n_texts - 30} more.")

    if results.text_groups:
        multi = [g for g in results.text_groups if len(g) > 1]
        if multi:
            parts.append(f"{len(multi)} multi-line label groups detected:")
            for grp in multi[:10]:
                combined = " / ".join(t["text"] for t in grp)
                parts.append(f'  - "{combined}"')

    # -- Shapes ---------------------------------------------------------------
    zones = results.shapes.get("zones", [])
    devices = results.shapes.get("devices", [])
    n_icons = len(results.icons)
    parts.append(
        f"Shapes: {len(zones)} zones, {len(devices)} device boxes, "
        f"{n_icons} icons detected."
    )

    return "\n".join(parts)
