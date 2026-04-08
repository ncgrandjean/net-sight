"""Image enhancement pipeline for network diagram preprocessing."""

import cv2
import numpy as np


def upscale(img: np.ndarray, factor: int) -> np.ndarray:
    """Upscale image using Lanczos interpolation."""
    if factor <= 1:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (w * factor, h * factor), interpolation=cv2.INTER_LANCZOS4)


def enhance_lines(img: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    """Thicken thin lines via morphological dilation."""
    if iterations <= 0:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(binary, kernel, iterations=iterations)

    if img.ndim == 3:
        # Apply dilation as a mask: darken pixels where lines were thickened
        mask = (dilated > 0) & (binary == 0)  # newly added pixels only
        result = img.copy()
        result[mask] = 0  # black for new line pixels
        return result
    else:
        return cv2.bitwise_not(dilated)


def enhance_contrast(
    img: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8
) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    if img.ndim == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        return clahe.apply(img)


def sharpen(img: np.ndarray, amount: float = 1.5) -> np.ndarray:
    """Sharpen via unsharp masking."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)


def apply_clahe(img, clip_limit=2.0, grid_size=8):
    """Convenience alias for :func:`enhance_contrast` matching the spec name."""
    return enhance_contrast(img, clip_limit=clip_limit, grid_size=grid_size)
