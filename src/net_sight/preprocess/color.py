"""Colour-channel separation via HSV masking for cable-type isolation."""

import cv2
import numpy as np


# Default colour ranges (HSV).  Hue is 0-179 in OpenCV.
# Each entry: (label, lower_hsv, upper_hsv)
DEFAULT_COLOR_RANGES = [
    ("red_low",    (0,   70, 50), (10,  255, 255)),
    ("red_high",   (170, 70, 50), (179, 255, 255)),
    ("orange",     (11,  70, 50), (25,  255, 255)),
    ("yellow",     (26,  70, 50), (35,  255, 255)),
    ("green",      (36,  70, 50), (85,  255, 255)),
    ("cyan",       (86,  70, 50), (100, 255, 255)),
    ("blue",       (101, 70, 50), (130, 255, 255)),
    ("purple",     (131, 70, 50), (160, 255, 255)),
    ("magenta",    (161, 70, 50), (169, 255, 255)),
]


def extract_color_channels(img, color_ranges=None):
    """Separate an image into masks for each dominant line colour.

    Each colour range is defined as an HSV interval.  The function
    returns a dict mapping the colour label to its binary mask
    (255 = colour present, 0 = absent).

    Red wraps around the hue wheel, so two ranges ("red_low" and
    "red_high") are provided by default and merged into a single
    "red" channel.

    Parameters
    ----------
    img : np.ndarray
        BGR colour image.
    color_ranges : list[tuple] | None
        Custom colour ranges as ``(label, lower_hsv, upper_hsv)`` tuples.
        Falls back to :data:`DEFAULT_COLOR_RANGES` when *None*.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of colour label to binary mask (uint8, same H x W as input).
    """
    if img.ndim != 3:
        print("[color] WARNING: grayscale image, returning empty channels")
        return {}

    if color_ranges is None:
        color_ranges = DEFAULT_COLOR_RANGES

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    raw_masks = {}
    for label, lower, upper in color_ranges:
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_np, upper_np)
        raw_masks[label] = mask

    # Merge the two red ranges into a single "red" channel.
    channels = {}
    red_low = raw_masks.pop("red_low", None)
    red_high = raw_masks.pop("red_high", None)
    if red_low is not None or red_high is not None:
        combined = np.zeros(img.shape[:2], dtype=np.uint8)
        if red_low is not None:
            combined = cv2.bitwise_or(combined, red_low)
        if red_high is not None:
            combined = cv2.bitwise_or(combined, red_high)
        channels["red"] = combined

    # Copy remaining channels as-is.
    for label, mask in raw_masks.items():
        channels[label] = mask

    # Filter out empty channels.
    channels = {k: v for k, v in channels.items() if cv2.countNonZero(v) > 0}

    print(f"[color] Extracted {len(channels)} colour channel(s): {list(channels.keys())}")
    return channels
