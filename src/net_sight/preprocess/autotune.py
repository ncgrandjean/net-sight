"""Auto-calibration of preprocessing parameters based on image characteristics."""

import cv2
import numpy as np


def _skeletonize(binary):
    """Morphological skeletonization using iterative erosion and opening."""
    skel = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    remaining = binary.copy()
    while True:
        eroded = cv2.erode(remaining, element)
        opened = cv2.dilate(eroded, element)
        diff = cv2.subtract(remaining, opened)
        skel = cv2.bitwise_or(skel, diff)
        remaining = eroded
        if cv2.countNonZero(remaining) == 0:
            break
    return skel


def analyze_image(img):
    """Analyze image characteristics for preprocessing calibration.

    Returns a dict with the following keys:

    - ``mean_line_thickness``: average line width in pixels (skeleton method).
    - ``line_density``: ratio of foreground (ink) pixels to total area.
    - ``contrast_score``: standard deviation of grayscale intensities.
    - ``text_density``: estimated ratio of text-like pixels to total area.
    - ``num_distinct_colors``: number of distinct colour clusters on lines.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    # Binarize: lines are dark pixels on light background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    total_pixels = binary.shape[0] * binary.shape[1]
    fg_pixels = int(np.count_nonzero(binary))
    line_density = fg_pixels / total_pixels if total_pixels > 0 else 0.0

    # Mean line thickness via morphological skeletonization (no opencv-contrib needed)
    skeleton = _skeletonize(binary)
    skeleton_pixels = int(np.count_nonzero(skeleton))
    mean_line_thickness = fg_pixels / skeleton_pixels if skeleton_pixels > 0 else 1.0

    # Contrast score: std dev of grayscale intensities
    contrast_score = float(np.std(gray))

    # Text density: ratio of small connected components (text-like) to total foreground
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
        widths = stats[1:, cv2.CC_STAT_WIDTH]
        heights = stats[1:, cv2.CC_STAT_HEIGHT]
        aspect = np.where(heights > 0, widths / heights, 0)
        # Text components: small area, reasonable aspect ratio
        text_mask = (areas < total_pixels * 0.005) & (aspect > 0.1) & (aspect < 10)
        text_pixels = int(np.sum(areas[text_mask]))
        text_density = text_pixels / total_pixels
    else:
        text_density = 0.0

    # Distinct colors on line pixels (from original color image)
    if img.ndim == 3:
        line_pixels_coords = np.column_stack(np.where(binary > 0))
        if len(line_pixels_coords) > 0:
            colors = img[line_pixels_coords[:, 0], line_pixels_coords[:, 1]].astype(np.float32)
            # Subsample for speed if too many pixels
            if len(colors) > 10_000:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(colors), 10_000, replace=False)
                colors = colors[idx]
            # K-means to find distinct color clusters
            best_k = 1
            for k in range(2, 13):
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
                compactness, _, _ = cv2.kmeans(
                    colors, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
                )
                # Elbow heuristic: stop if compactness reduction < 20%
                if k == 2:
                    prev_compactness = compactness
                    best_k = k
                else:
                    ratio = compactness / prev_compactness if prev_compactness > 0 else 1.0
                    if ratio > 0.8:
                        break
                    prev_compactness = compactness
                    best_k = k
            num_distinct_colors = best_k
        else:
            num_distinct_colors = 0
    else:
        num_distinct_colors = 1

    return {
        "mean_line_thickness": float(mean_line_thickness),
        "line_density": float(line_density),
        "contrast_score": float(contrast_score),
        "text_density": float(text_density),
        "num_distinct_colors": int(num_distinct_colors),
    }


# Public alias expected by the pipeline module.
analyze_image_characteristics = analyze_image


def compute_params(analysis):
    """Derive optimal preprocessing parameters from image analysis.

    Returns a dict consumed by :func:`pipeline.preprocess`:

    - ``upscale_factor``
    - ``morph_kernel_size``, ``morph_iterations``
    - ``close_gaps_kernel`` (0 = disabled)
    - ``clahe_enabled``, ``clahe_clip_limit``, ``clahe_grid_size``
    - ``sharpen_enabled``, ``sharpen_amount``
    - ``tile_count`` (for CLAHE grid or downstream tiling)
    """
    thickness = analysis["mean_line_thickness"]
    density = analysis["line_density"]
    contrast = analysis["contrast_score"]
    text_density = analysis.get("text_density", 0.0)

    # ------------------------------------------------------------------
    # Upscale: enlarge when lines are dangerously thin
    # ------------------------------------------------------------------
    upscale_factor = 1 if thickness >= 3.0 else 2

    # ------------------------------------------------------------------
    # Morphological dilation: thicken thin lines
    # ------------------------------------------------------------------
    morph_kernel_size = 3

    if thickness >= 4.0:
        morph_iterations = 0
    elif thickness >= 2.0:
        morph_iterations = 1
    else:
        morph_iterations = 2

    # ------------------------------------------------------------------
    # Gap closing: connect broken segments
    # ------------------------------------------------------------------
    # Enable gap closing for very thin lines or sparse diagrams.
    if thickness < 2.0 or density < 0.03:
        close_gaps_kernel = 3
    else:
        close_gaps_kernel = 0  # disabled

    # ------------------------------------------------------------------
    # CLAHE: boost contrast when the image is washed-out
    # ------------------------------------------------------------------
    clahe_enabled = contrast < 50.0
    clahe_clip_limit = 2.0
    clahe_grid_size = 8

    # ------------------------------------------------------------------
    # Sharpening
    # ------------------------------------------------------------------
    sharpen_enabled = True
    # Reduce sharpening when text is dense to avoid ringing artefacts.
    sharpen_amount = 1.0 if text_density > 0.05 else 1.5

    # ------------------------------------------------------------------
    # CLAHE tile count (also useful for downstream tiling hints)
    # ------------------------------------------------------------------
    if density > 0.15:
        tile_count = 16
    elif density > 0.08:
        tile_count = 12
    elif density > 0.03:
        tile_count = 8
    else:
        tile_count = 4

    return {
        "upscale_factor": upscale_factor,
        "morph_kernel_size": morph_kernel_size,
        "morph_iterations": morph_iterations,
        "close_gaps_kernel": close_gaps_kernel,
        "clahe_enabled": clahe_enabled,
        "clahe_clip_limit": clahe_clip_limit,
        "clahe_grid_size": clahe_grid_size,
        "sharpen_enabled": sharpen_enabled,
        "sharpen_amount": sharpen_amount,
        "tile_count": tile_count,
    }
