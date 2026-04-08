"""Main analysis pipeline: preprocess -> tile -> CV -> VLM -> merge -> output."""

from __future__ import annotations

import asyncio
import os
import time

import cv2
import numpy as np

from net_sight.preprocess.autotune import analyze_image, compute_params
from net_sight.preprocess.enhance import enhance_contrast, sharpen, upscale
from net_sight.preprocess.morphology import close_gaps, dilate_lines
from net_sight.tiling.grid import (
    Tile,
    compute_grid,
    create_global_view,
    get_adjacent_pairs,
    split_into_tiles,
)
from net_sight.cv.lines import classify_line_types, detect_lines
from net_sight.cv.ocr import extract_texts
from net_sight.cv.shapes import detect_shapes
from net_sight.cv.colors import analyze_line_colors
from net_sight.analyze.ollama_client import OllamaClient
from net_sight.analyze.passes import run_all_passes
from net_sight.merge.consolidate import merge_tile_results
from net_sight.output.markdown import format_cv_summary, format_report

# ----- Configuration (edit here) -----
WORKERS = 4
MODEL = "qwen3-vl:8b"
OLLAMA_URL = "http://localhost:11434"
TIMEOUT = 300
# --------------------------------------


def run(image_path: str) -> str:
    """Run the full analysis pipeline on a network diagram image.

    Returns the path to the generated markdown report.
    """
    t0 = time.time()
    print(f"[net-sight] Analyzing: {image_path}")

    # --- Load image ---
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]
    print(f"[net-sight] Image size: {w}x{h}")

    # --- Step 1: Auto-tune and preprocess ---
    print("[net-sight] Step 1/5: Analyzing image characteristics...")
    characteristics = analyze_image(img)
    params = compute_params(characteristics)
    print(f"[net-sight]   Auto-tuned params: {params}")

    print("[net-sight] Step 1/5: Preprocessing...")
    enhanced = _apply_preprocessing(img, params)

    # --- Step 2: Tiling ---
    print("[net-sight] Step 2/5: Tiling...")
    eh, ew = enhanced.shape[:2]
    max_tiles = params.get("tile_count", 16)
    rows, cols = compute_grid(eh, ew, target_tile_size=1024, overlap=0.25, max_tiles=max_tiles)
    tiles = split_into_tiles(enhanced, rows, cols, overlap=0.25)
    global_view = create_global_view(enhanced)
    adjacent_pairs = get_adjacent_pairs(tiles, rows, cols)
    print(f"[net-sight]   {len(tiles)} tiles, {len(adjacent_pairs)} adjacent pairs")

    # --- Step 3: CV augmentation ---
    print("[net-sight] Step 3/5: Computer vision analysis...")
    cv_data = _run_cv(enhanced, tiles, adjacent_pairs)

    # --- Step 4: VLM analysis ---
    print(f"[net-sight] Step 4/5: VLM analysis ({MODEL}, {WORKERS} workers)...")
    vlm_results = asyncio.run(
        _run_vlm(global_view, tiles, adjacent_pairs, enhanced, cv_data)
    )

    # --- Step 5: Merge and output ---
    print("[net-sight] Step 5/5: Merging results...")

    # Build tile result dicts for merge (add positional info + VLM text)
    tile_result_dicts = []
    for tile, result_text in zip(tiles, vlm_results["tiles"]):
        tile_result_dicts.append({
            "row": tile.row,
            "col": tile.col,
            "x": tile.x,
            "y": tile.y,
            "result": result_text,
        })

    merged = merge_tile_results(
        vlm_results["global"],
        tile_result_dicts,
        vlm_results["cross_tile"],
    )

    elapsed = round(time.time() - t0, 1)
    meta = {
        "Source": os.path.basename(image_path),
        "Image size": f"{w}x{h}",
        "Upscale": f"x{params['upscale_factor']}",
        "Dilation": f"kernel={params['morph_kernel_size']} iter={params['morph_iterations']}",
        "Tiles": f"{len(tiles)} ({rows}x{cols})",
        "Model": MODEL,
        "Workers": WORKERS,
        "Duration": f"{elapsed}s",
    }

    cv_summary = format_cv_summary(
        lines_count=cv_data["lines_count"],
        texts_count=cv_data["texts_count"],
        shapes_count=cv_data["shapes_count"],
        color_clusters=cv_data["color_clusters"],
    )

    report = format_report(image_path, merged, cv_summary, meta)

    # Write output
    output_path = os.path.splitext(image_path)[0] + ".md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[net-sight] Done in {elapsed}s -> {output_path}")
    return output_path


def _apply_preprocessing(img: np.ndarray, params: dict) -> np.ndarray:
    """Apply the preprocessing steps using autotune params."""
    result = upscale(img, params.get("upscale_factor", 1))

    if params.get("morph_iterations", 0) > 0:
        result = dilate_lines(
            result,
            kernel_size=params.get("morph_kernel_size", 3),
            iterations=params["morph_iterations"],
        )

    if params.get("close_gaps_kernel", 0) > 0:
        result = close_gaps(result, kernel_size=params["close_gaps_kernel"])

    if params.get("clahe_enabled", False):
        result = enhance_contrast(
            result,
            clip_limit=params.get("clahe_clip_limit", 2.0),
            grid_size=params.get("clahe_grid_size", 8),
        )

    if params.get("sharpen_enabled", True):
        result = sharpen(result, amount=params.get("sharpen_amount", 1.5))

    return result


def _extract_overlap_image(
    full_img: np.ndarray, tile_a: Tile, tile_b: Tile
) -> np.ndarray:
    """Crop the overlap region between two adjacent tiles from the full image."""
    if tile_a.row == tile_b.row:
        # Horizontal adjacency
        ox = tile_b.x
        oy = max(tile_a.y, tile_b.y)
        ow = (tile_a.x + tile_a.width) - tile_b.x
        oh = min(tile_a.y + tile_a.height, tile_b.y + tile_b.height) - oy
    else:
        # Vertical adjacency
        ox = max(tile_a.x, tile_b.x)
        oy = tile_b.y
        ow = min(tile_a.x + tile_a.width, tile_b.x + tile_b.width) - ox
        oh = (tile_a.y + tile_a.height) - tile_b.y

    # Clamp to valid bounds
    ow = max(1, ow)
    oh = max(1, oh)
    return full_img[oy : oy + oh, ox : ox + ow].copy()


def _tile_to_meta(tile: Tile) -> dict:
    """Convert a Tile dataclass to a metadata dict for the VLM passes."""
    return {
        "row": tile.row,
        "col": tile.col,
        "x": tile.x,
        "y": tile.y,
        "w": tile.width,
        "h": tile.height,
    }


def _run_cv(
    enhanced: np.ndarray, tiles: list[Tile], adjacent_pairs: list[tuple[Tile, Tile]]
) -> dict:
    """Run CV augmentation on the full image and collect stats."""
    lines = detect_lines(enhanced)
    lines = classify_line_types(lines, enhanced)
    texts = extract_texts(enhanced)
    shapes = detect_shapes(enhanced)
    color_info = analyze_line_colors(enhanced, lines)

    # Build per-tile CV context strings for VLM prompts
    tile_cv_contexts = []
    for tile in tiles:
        tile_lines = detect_lines(tile.image, min_length=15)
        tile_texts_count = len([
            t for t in texts
            if tile.x <= (t["bbox"][0] + t["bbox"][2]) / 2 <= tile.x + tile.width
            and tile.y <= (t["bbox"][1] + t["bbox"][3]) / 2 <= tile.y + tile.height
        ])
        ctx = (
            f"CV analysis of this zone detected {len(tile_lines)} line segments "
            f"and approximately {tile_texts_count} text labels. "
        )
        if tile_lines:
            horiz = sum(1 for ln in tile_lines if ln["angle"] < 30)
            orient = "horizontal" if horiz > len(tile_lines) / 2 else "mixed"
            ctx += f"Line orientations: mostly {orient}. "
        tile_cv_contexts.append(ctx)

    # Build cross-tile CV contexts
    pair_cv_contexts = []
    for _ in adjacent_pairs:
        pair_cv_contexts.append(
            "Focus on connections crossing the boundary between these adjacent zones."
        )

    return {
        "lines_count": len(lines),
        "texts_count": len(texts),
        "shapes_count": len(shapes),
        "color_clusters": len(color_info.get("clusters", [])),
        "tile_cv_contexts": tile_cv_contexts,
        "pair_cv_contexts": pair_cv_contexts,
        "all_texts": texts,
        "all_lines": lines,
    }


async def _run_vlm(
    global_view: np.ndarray,
    tiles: list[Tile],
    adjacent_pairs: list[tuple[Tile, Tile]],
    full_img: np.ndarray,
    cv_data: dict,
) -> dict:
    """Run all VLM passes asynchronously."""
    client = OllamaClient(model=MODEL, base_url=OLLAMA_URL, timeout=TIMEOUT)

    # Convert Tile objects to (meta_dict, image) tuples expected by passes.py
    tile_tuples = [(_tile_to_meta(t), t.image) for t in tiles]

    # Extract overlap images for cross-tile pass
    pair_meta = [(_tile_to_meta(a), _tile_to_meta(b)) for a, b in adjacent_pairs]
    pair_images = [_extract_overlap_image(full_img, a, b) for a, b in adjacent_pairs]

    return await run_all_passes(
        client=client,
        global_view=global_view,
        tiles=tile_tuples,
        adjacent_pairs=pair_meta,
        pair_images=pair_images,
        cv_contexts_tiles=cv_data["tile_cv_contexts"],
        cv_contexts_pairs=cv_data.get("pair_cv_contexts", []),
        workers=WORKERS,
    )
