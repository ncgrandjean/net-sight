"""Morphological operations for thin-line enhancement in network diagrams."""

import cv2
import numpy as np


def dilate_lines(img, kernel_size=3, iterations=1):
    """Thicken thin lines via morphological dilation.

    Works on both BGR colour images and single-channel grayscale images.
    For colour images the dilation is computed on a binarised version
    (Otsu threshold on the inverse) and newly-expanded pixels are painted
    black on the original image so that colour information is preserved
    everywhere else.

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR or grayscale).
    kernel_size : int
        Size of the structuring element (ellipse).
    iterations : int
        Number of dilation passes.  0 returns the image unchanged.

    Returns
    -------
    np.ndarray
        Image with thickened lines.
    """
    if iterations <= 0:
        return img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(binary, kernel, iterations=iterations)

    if img.ndim == 3:
        # Only darken the *newly added* pixels so the original colours are kept.
        mask = (dilated > 0) & (binary == 0)
        result = img.copy()
        result[mask] = 0
        return result

    return cv2.bitwise_not(dilated)


def close_gaps(img, kernel_size=3):
    """Close small gaps between line segments using morphological closing.

    Morphological closing (dilation then erosion) connects nearby
    endpoints without noticeably thickening the lines.

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR or grayscale).
    kernel_size : int
        Size of the structuring element used for closing.

    Returns
    -------
    np.ndarray
        Image with gaps closed.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    if img.ndim == 3:
        # Paint newly-connected pixels black on the colour image.
        new_pixels = (closed > 0) & (binary == 0)
        result = img.copy()
        result[new_pixels] = 0
        return result

    return cv2.bitwise_not(closed)
