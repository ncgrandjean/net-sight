"""Preprocessing module: auto-calibration, morphology, colour separation, and image enhancement."""

from .autotune import analyze_image, analyze_image_characteristics, compute_params
from .color import extract_color_channels
from .enhance import apply_clahe, enhance_contrast, enhance_lines, sharpen, upscale
from .morphology import close_gaps, dilate_lines
from .pipeline import preprocess

__all__ = [
    "analyze_image",
    "analyze_image_characteristics",
    "apply_clahe",
    "close_gaps",
    "compute_params",
    "dilate_lines",
    "enhance_contrast",
    "enhance_lines",
    "extract_color_channels",
    "preprocess",
    "sharpen",
    "upscale",
]
