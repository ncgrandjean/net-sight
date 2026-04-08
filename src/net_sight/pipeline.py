"""Main analysis pipeline: preprocess -> tile -> CV -> VLM -> merge -> output."""

from __future__ import annotations

import asyncio
import gc
import os
import time
import traceback

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
from net_sight.cv.colors import cluster_colors
from net_sight.analyze.ollama_client import OllamaClient
from net_sight.analyze.passes import run_all_passes
from net_sight.merge.consolidate import merge_tile_results
from net_sight.output.markdown import format_cv_summary, format_report

# ----- Configuration (edit here) -----
WORKERS = 1
MODEL = "qwen3-vl:8b"
OLLAMA_URL = "http://localhost:11434"
TIMEOUT = 3600
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
    del img  # Free original image
    gc.collect()

    # --- Step 2: Tiling ---
    print("[net-sight] Step 2/5: Tiling...")
    eh, ew = enhanced.shape[:2]
    max_tiles = params.get("tile_count", 16)
    rows, cols = compute_grid(eh, ew, target_tile_size=1024, overlap=0.25, max_tiles=max_tiles)
    tiles = split_into_tiles(enhanced, rows, cols, overlap=0.25)
    global_view = create_global_view(enhanced)
    adjacent_pairs = get_adjacent_pairs(tiles, rows, cols)
    print(f"[net-sight]   {len(tiles)} tiles, {len(adjacent_pairs)} adjacent pairs")

    # Extract overlap images before freeing the full enhanced image
    pair_images = [_extract_overlap_image(enhanced, a, b) for a, b in adjacent_pairs]
    del enhanced  # Free the large upscaled image (~200 Mo)
    gc.collect()

    # --- Step 3: CV augmentation (per-tile, not on the huge full image) ---
    print("[net-sight] Step 3/5: Computer vision analysis...")
    cv_data = _run_cv_per_tile(tiles, adjacent_pairs)

    # --- Step 4: VLM analysis ---
    print(f"[net-sight] Step 4/5: VLM analysis ({MODEL}, {WORKERS} workers)...")
    vlm_results = asyncio.run(
        _run_vlm(global_view, tiles, adjacent_pairs, pair_images, cv_data)
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


def _cluster_existing_colors(lines: list[dict]) -> dict:
    """Cluster line colors using color_rgb already stored in each line dict."""
    colors = []
    for ln in lines:
        c = ln.get("color_rgb") or ln.get("color")
        if c:
            colors.append(c)
    if not colors:
        return {"clusters": []}
    clusters = cluster_colors(colors)
    result = []
    for label, rgb in clusters.items():
        count = sum(1 for c in colors if c == rgb)
        result.append({
            "color_rgb": rgb,
            "count": count,
            "percentage": round(count / len(colors) * 100, 2),
        })
    result.sort(key=lambda c: c["count"], reverse=True)
    return {"clusters": result, "total_lines": len(lines)}


def _run_cv_per_tile(
    tiles: list[Tile], adjacent_pairs: list[tuple[Tile, Tile]]
) -> dict:
    """Run CV augmentation per tile instead of on the full image.

    This avoids loading the full upscaled image into OCR/line detection,
    saving several GB of RAM.
    """
    total_lines = 0
    total_texts = 0
    total_shapes = 0
    all_lines: list[dict] = []
    all_texts: list[dict] = []

    tile_cv_contexts = []

    for i, tile in enumerate(tiles):
        print(f"[net-sight]   CV: tile {i + 1}/{len(tiles)}...")

        tile_lines = detect_lines(tile.image, min_length=15)
        line_groups = classify_line_types(tile_lines, tile.image)
        enriched = [ln for group in line_groups.values() for ln in group]

        tile_texts = extract_texts(tile.image)
        tile_shapes = detect_shapes(tile.image)

        # Remap coordinates to global image frame
        for ln in enriched:
            ln["x1"] += tile.x
            ln["y1"] += tile.y
            ln["x2"] += tile.x
            ln["y2"] += tile.y

        for t in tile_texts:
            t["x"] += tile.x
            t["y"] += tile.y

        all_lines.extend(enriched)
        all_texts.extend(tile_texts)
        total_lines += len(enriched)
        total_texts += len(tile_texts)
        total_shapes += len(tile_shapes)

        # Build CV context for this tile's VLM prompt
        ctx = (
            f"CV analysis of this zone detected {len(tile_lines)} line segments "
            f"and {len(tile_texts)} text labels. "
        )
        if tile_lines:
            horiz = sum(1 for ln in tile_lines if ln["angle"] < 30)
            orient = "horizontal" if horiz > len(tile_lines) / 2 else "mixed"
            ctx += f"Line orientations: mostly {orient}. "
        if tile_texts:
            labels = [t["text"] for t in tile_texts[:10]]
            ctx += f"Detected labels include: {', '.join(labels)}. "
        tile_cv_contexts.append(ctx)

    # Color clustering from already-extracted line colors (no image needed)
    color_info = _cluster_existing_colors(all_lines)

    # Build cross-tile CV contexts
    pair_cv_contexts = []
    for _ in adjacent_pairs:
        pair_cv_contexts.append(
            "Focus on connections crossing the boundary between these adjacent zones."
        )

    return {
        "lines_count": total_lines,
        "texts_count": total_texts,
        "shapes_count": total_shapes,
        "color_clusters": len(color_info.get("clusters", [])),
        "tile_cv_contexts": tile_cv_contexts,
        "pair_cv_contexts": pair_cv_contexts,
        "all_texts": all_texts,
        "all_lines": all_lines,
    }


async def _run_vlm(
    global_view: np.ndarray,
    tiles: list[Tile],
    adjacent_pairs: list[tuple[Tile, Tile]],
    pair_images: list[np.ndarray],
    cv_data: dict,
) -> dict:
    """Run all VLM passes asynchronously."""
    client = OllamaClient(model=MODEL, base_url=OLLAMA_URL, timeout=TIMEOUT)

    # Convert Tile objects to (meta_dict, image) tuples expected by passes.py
    tile_tuples = [(_tile_to_meta(t), t.image) for t in tiles]

    # Build pair metadata
    pair_meta = [(_tile_to_meta(a), _tile_to_meta(b)) for a, b in adjacent_pairs]

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
