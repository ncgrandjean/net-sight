"""Grid-based image tiling with overlap for large network diagram analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Tile:
    """A single tile extracted from a larger image.

    Attributes:
        image: Pixel data for this tile (H x W x C numpy array).
        row: Row index in the tiling grid.
        col: Column index in the tiling grid.
        x: Left-edge X coordinate in the original image.
        y: Top-edge Y coordinate in the original image.
        width: Tile width in pixels.
        height: Tile height in pixels.
    """

    image: np.ndarray
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int


def create_global_view(img: np.ndarray, target_size: int = 1024) -> np.ndarray:
    """Resize *img* so its longest side equals *target_size*, preserving aspect ratio.

    Uses INTER_AREA for downscaling (anti-aliased) and INTER_LINEAR for upscaling.
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w = round(w * scale)
    new_h = round(h * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def compute_grid(
    img_height: int,
    img_width: int,
    target_tile_size: int = 1024,
    overlap: float = 0.25,
) -> tuple[int, int]:
    """Compute the optimal number of grid rows and columns.

    The grid is sized so that each tile is close to *target_tile_size* pixels on
    each side, with an *overlap* fraction shared between adjacent tiles.

    Returns:
        (rows, cols) tuple.
    """
    overlap_px = round(target_tile_size * overlap)
    stride = target_tile_size - overlap_px

    cols = max(1, math.ceil((img_width - overlap_px) / stride))
    rows = max(1, math.ceil((img_height - overlap_px) / stride))
    return rows, cols


def _axis_positions(
    length: int,
    n: int,
    overlap: float,
) -> list[tuple[int, int]]:
    """Compute (origin, size) pairs for *n* tiles along an axis of *length* pixels.

    Tiles are evenly spaced so that the requested *overlap* fraction is
    approximately respected. The first tile starts at 0 and the last tile ends at
    *length*.
    """
    if n == 1:
        return [(0, length)]

    tile_sz = round(length / (1 + (1 - overlap) * (n - 1)))
    tile_sz = max(1, min(tile_sz, length))
    step = (length - tile_sz) / (n - 1)

    positions: list[tuple[int, int]] = []
    for i in range(n):
        origin = round(step * i)
        origin = max(0, min(origin, length - tile_sz))
        actual_sz = min(tile_sz, length - origin)
        positions.append((origin, actual_sz))
    return positions


def split_into_tiles(
    img: np.ndarray,
    rows: int,
    cols: int,
    overlap: float = 0.25,
) -> list[Tile]:
    """Split *img* into a grid of tiles with the given overlap.

    Args:
        img: Source image (H x W x C or H x W).
        rows: Number of tile rows.
        cols: Number of tile columns.
        overlap: Fraction of tile size shared between adjacent tiles (0-1).

    Returns:
        Flat list of :class:`Tile` objects ordered row-major (row 0 col 0 first).
    """
    img_h, img_w = img.shape[:2]
    x_positions = _axis_positions(img_w, cols, overlap)
    y_positions = _axis_positions(img_h, rows, overlap)

    tiles: list[Tile] = []
    for r, (y, h) in enumerate(y_positions):
        for c, (x, w) in enumerate(x_positions):
            tile_img = img[y : y + h, x : x + w].copy()
            tiles.append(
                Tile(
                    image=tile_img,
                    row=r,
                    col=c,
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                )
            )
    return tiles


def get_adjacent_pairs(
    tiles: list[Tile],
    rows: int,
    cols: int,
) -> list[tuple[Tile, Tile]]:
    """Return pairs of adjacent tiles with their shared overlap region.

    Two tiles are adjacent if they differ by exactly one step horizontally or
    vertically. Each pair is yielded once (the tile with the smaller grid index
    comes first).

    Returns:
        List of (tile_a, tile_b) tuples for every horizontal or vertical
        adjacency. The overlap region between tile_a and tile_b can be computed
        from their positional attributes:

        * Horizontal pair (same row, cols differ by 1):
          overlap_x = tile_b.x
          overlap_y = max(tile_a.y, tile_b.y)
          overlap_w = (tile_a.x + tile_a.width) - tile_b.x
          overlap_h = min(tile_a.y + tile_a.height, tile_b.y + tile_b.height) - overlap_y

        * Vertical pair (same col, rows differ by 1):
          overlap_x = max(tile_a.x, tile_b.x)
          overlap_y = tile_b.y
          overlap_w = min(tile_a.x + tile_a.width, tile_b.x + tile_b.width) - overlap_x
          overlap_h = (tile_a.y + tile_a.height) - tile_b.y
    """
    index: dict[tuple[int, int], Tile] = {(t.row, t.col): t for t in tiles}
    pairs: list[tuple[Tile, Tile]] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    for (r, c), tile in index.items():
        for dr, dc in ((0, 1), (1, 0)):
            nr, nc = r + dr, c + dc
            if nr >= rows or nc >= cols:
                continue
            neighbor = index.get((nr, nc))
            if neighbor is None:
                continue
            key = ((r, c), (nr, nc))
            if key not in seen:
                seen.add(key)
                pairs.append((tile, neighbor))
    return pairs
