"""Preprocessing pipeline orchestrator for network diagram images."""

import cv2
import numpy as np

from .autotune import analyze_image_characteristics, compute_params
from .color import extract_color_channels
from .enhance import apply_clahe, sharpen, upscale
from .morphology import close_gaps, dilate_lines


def preprocess(img_path, return_color_channels=False):
    """Full preprocessing pipeline with automatic parameter calibration.

    Steps executed in order:
      1. Load image from disk.
      2. Analyse image characteristics (autotune).
      3. Compute optimal parameters.
      4. Upscale (Lanczos) if lines are too thin.
      5. Morphological dilation to thicken lines.
      6. Gap closing to reconnect broken segments.
      7. CLAHE contrast enhancement (if needed).
      8. Unsharp-mask sharpening.

    Parameters
    ----------
    img_path : str
        Filesystem path to the source PNG/JPEG image.
    return_color_channels : bool
        When *True*, also return a dict of per-colour binary masks
        extracted from the *original* (pre-enhanced) image.

    Returns
    -------
    np.ndarray
        Enhanced BGR image ready for VLM analysis.
    dict (optional)
        Colour channel masks, only when *return_color_channels* is True.
    """
    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    print(f"[pipeline] Loaded image {img_path}  ({img.shape[1]}x{img.shape[0]})")

    # ------------------------------------------------------------------
    # 2. Analyse
    # ------------------------------------------------------------------
    characteristics = analyze_image_characteristics(img)
    print("[pipeline] Image characteristics:")
    for key, val in characteristics.items():
        print(f"  {key}: {val}")

    # ------------------------------------------------------------------
    # 3. Compute parameters
    # ------------------------------------------------------------------
    params = compute_params(characteristics)
    print("[pipeline] Auto-tuned parameters:")
    for key, val in params.items():
        print(f"  {key}: {val}")

    # ------------------------------------------------------------------
    # 4. Upscale
    # ------------------------------------------------------------------
    result = upscale(img, params["upscale_factor"])
    if params["upscale_factor"] > 1:
        print(f"[pipeline] Upscaled x{params['upscale_factor']}  "
              f"-> {result.shape[1]}x{result.shape[0]}")

    # ------------------------------------------------------------------
    # 5. Morphological dilation
    # ------------------------------------------------------------------
    if params["morph_iterations"] > 0:
        result = dilate_lines(
            result,
            kernel_size=params["morph_kernel_size"],
            iterations=params["morph_iterations"],
        )
        print(f"[pipeline] Dilated lines  kernel={params['morph_kernel_size']}  "
              f"iterations={params['morph_iterations']}")

    # ------------------------------------------------------------------
    # 6. Gap closing
    # ------------------------------------------------------------------
    if params.get("close_gaps_kernel", 0) > 0:
        result = close_gaps(result, kernel_size=params["close_gaps_kernel"])
        print(f"[pipeline] Closed gaps  kernel={params['close_gaps_kernel']}")

    # ------------------------------------------------------------------
    # 7. CLAHE
    # ------------------------------------------------------------------
    if params.get("clahe_enabled", False):
        result = apply_clahe(
            result,
            clip_limit=params.get("clahe_clip_limit", 2.0),
            grid_size=params.get("clahe_grid_size", 8),
        )
        print("[pipeline] Applied CLAHE")

    # ------------------------------------------------------------------
    # 8. Sharpening
    # ------------------------------------------------------------------
    if params.get("sharpen_enabled", True):
        amount = params.get("sharpen_amount", 1.5)
        result = sharpen(result, amount=amount)
        print(f"[pipeline] Sharpened  amount={amount}")

    print("[pipeline] Preprocessing complete.")

    # ------------------------------------------------------------------
    # Optional: colour channel extraction (on the original image)
    # ------------------------------------------------------------------
    if return_color_channels:
        channels = extract_color_channels(img)
        return result, channels

    return result
