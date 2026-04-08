"""Computer-vision augmentation pipeline for network diagrams."""

from net_sight.cv.colors import (
    analyze_line_colors,
    build_color_legend,
    cluster_colors,
    extract_line_colors,
    match_legend_colors,
)
from net_sight.cv.lines import classify_line_types, count_connections, detect_lines
from net_sight.cv.ocr import (
    extract_text,
    extract_text_in_region,
    extract_texts,
    extract_texts_from_tile,
    get_reader,
    group_text_by_proximity,
)
from net_sight.cv.pipeline import CVResults, format_cv_context, run_cv_analysis, run_cv_on_tile
from net_sight.cv.shapes import (
    classify_shapes,
    detect_icons,
    detect_rectangles,
    detect_shapes,
)

__all__ = [
    "CVResults",
    "analyze_line_colors",
    "build_color_legend",
    "classify_line_types",
    "classify_shapes",
    "cluster_colors",
    "count_connections",
    "detect_icons",
    "detect_lines",
    "detect_rectangles",
    "detect_shapes",
    "extract_line_colors",
    "extract_text",
    "extract_text_in_region",
    "extract_texts",
    "extract_texts_from_tile",
    "format_cv_context",
    "get_reader",
    "group_text_by_proximity",
    "match_legend_colors",
    "run_cv_analysis",
    "run_cv_on_tile",
]
