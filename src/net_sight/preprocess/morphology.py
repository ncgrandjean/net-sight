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

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    if img.ndim == 3:
        # Erode the colour image directly: cv2.erode takes the per-channel
        # minimum over the neighbourhood, which spreads dark (line) pixels
        # into the bright (background) while preserving original colours.
        return cv2.erode(img, kernel, iterations=iterations)

    gray = img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dilated = cv2.dilate(binary, kernel, iterations=iterations)
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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    if img.ndim == 3:
        # Lines are dark foreground on a bright background.  To close gaps
        # we erode first (spreads dark lines, bridging gaps) then dilate
        # (shrinks them back to roughly original thickness).  This is the
        # inverse of standard morphological closing because the objects of
        # interest are dark, not bright.
        eroded = cv2.erode(img, kernel)
        return cv2.dilate(eroded, kernel)

    gray = img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return cv2.bitwise_not(closed)
