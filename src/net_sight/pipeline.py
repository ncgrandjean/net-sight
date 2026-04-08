"""Main analysis pipeline: preprocess -> tile -> CV -> VLM -> merge -> output."""

from __future__ import annotations

import asyncio
import gc
import os
import time

import cv2
import numpy as np

from net_sight.preprocess.autotune import analyze_image, compute_params
from net_sight.preprocess.enhance import enhance_contrast, sharpen
from net_sight.preprocess.morphology import close_gaps, dilate_lines
from net_sight.tiling.grid import (
    Tile,
    compute_grid,
    create_global_view,
    split_into_tiles,
)
from net_sight.cv.lines import classify_line_types, detect_lines
from net_sight.cv.ocr import extract_texts
from net_sight.cv.shapes import detect_shapes
from net_sight.cv.colors import cluster_colors
from net_sight.analyze.ollama_client import OllamaClient
from net_sight.analyze.passes import run_global_pass, run_tile_pass
from net_sight.analyze.prompts import GLOBAL_PROMPT, TILE_PROMPT, format_prompt
from net_sight.merge.consolidate import merge_tile_results
from net_sight.output.markdown import format_cv_summary, format_report

# ----- Configuration (edit here) -----
WORKERS = 1
MODEL = "qwen3-vl:8b"
OLLAMA_URL = "http://localhost:11434"
TIMEOUT = 3600
VLM_TILE_SIZE = 1024  # Max tile size sent to VLM
# --------------------------------------


def run(image_path: str, debug: bool = False) -> str:
    """Run the full analysis pipeline on a network diagram image.

    Returns the path to the generated markdown report.
    """
    t0 = time.time()
    print(f"[net-sight] Analyzing: {image_path}")

    # Debug output directory
    debug_dir = None
    if debug:
        debug_dir = os.path.join(os.path.dirname(image_path) or ".", "debug")
        os.makedirs(debug_dir, exist_ok=True)
        print(f"[net-sight] Debug mode: saving intermediates to {debug_dir}/")

    # --- Load image ---
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]
    print(f"[net-sight] Image size: {w}x{h}")

    if debug:
        cv2.imwrite(os.path.join(debug_dir, "01_original.png"), img)

    # --- Step 1: Preprocess (no upscale, dilation only) ---
    print("[net-sight] Step 1/4: Analyzing image characteristics...")
    characteristics = analyze_image(img)
    params = compute_params(characteristics)
    # Force no upscale: dilation alone handles thin lines
    params["upscale_factor"] = 1
    print(f"[net-sight]   Line thickness: {characteristics['mean_line_thickness']:.1f}px")
    print(f"[net-sight]   Dilation: kernel={params['morph_kernel_size']} iter={params['morph_iterations']}")

    print("[net-sight] Step 1/4: Preprocessing...")
    enhanced = _apply_preprocessing(img, params, debug_dir)
    del img
    gc.collect()

    # --- Step 2: Tiling ---
    print("[net-sight] Step 2/4: Tiling...")
    eh, ew = enhanced.shape[:2]
    max_tiles = params.get("tile_count", 16)
    rows, cols = compute_grid(eh, ew, target_tile_size=1024, overlap=0.25, max_tiles=max_tiles)
    tiles = split_into_tiles(enhanced, rows, cols, overlap=0.25)
    global_view = create_global_view(enhanced)
    print(f"[net-sight]   {len(tiles)} tiles ({rows}x{cols})")

    if debug:
        cv2.imwrite(os.path.join(debug_dir, "05_global_view.png"), global_view)
        for tile in tiles:
            name = f"05_tile_{tile.row}_{tile.col}.png"
            cv2.imwrite(os.path.join(debug_dir, name), tile.image)

    del enhanced
    gc.collect()

    # --- Step 3: CV augmentation (per-tile) ---
    print("[net-sight] Step 3/4: Computer vision analysis...")
    cv_data = _run_cv_per_tile(tiles)

    # --- Step 4: VLM analysis (global + tiles only, no cross-tile pass) ---
    print(f"[net-sight] Step 4/4: VLM analysis ({MODEL})...")
    print(f"[net-sight]   1 global + {len(tiles)} tiles = {1 + len(tiles)} VLM calls")
    vlm_results = asyncio.run(_run_vlm(global_view, tiles, cv_data))

    # --- Output ---
    print("[net-sight] Generating report...")

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
        [],  # No cross-tile pass
    )

    elapsed = round(time.time() - t0, 1)
    meta = {
        "Source": os.path.basename(image_path),
        "Image size": f"{w}x{h}",
        "Dilation": f"kernel={params['morph_kernel_size']} iter={params['morph_iterations']}",
        "Tiles": f"{len(tiles)} ({rows}x{cols})",
        "Model": MODEL,
        "Duration": f"{elapsed}s",
    }

    cv_summary = format_cv_summary(
        lines_count=cv_data["lines_count"],
        texts_count=cv_data["texts_count"],
        shapes_count=cv_data["shapes_count"],
        color_clusters=cv_data["color_clusters"],
    )

    report = format_report(image_path, merged, cv_summary, meta)

    output_path = os.path.splitext(image_path)[0] + ".md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[net-sight] Done in {elapsed}s -> {output_path}")
    return output_path


# -----------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------

def _apply_preprocessing(
    img: np.ndarray, params: dict, debug_dir: str | None = None
) -> np.ndarray:
    """Apply preprocessing steps. No upscale, dilation handles thin lines."""
    result = img.copy()
    step = 1

    if params.get("morph_iterations", 0) > 0:
        step += 1
        result = dilate_lines(
            result,
            kernel_size=params.get("morph_kernel_size", 3),
            iterations=params["morph_iterations"],
        )
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, f"0{step}_dilated.png"), result)

    if params.get("close_gaps_kernel", 0) > 0:
        step += 1
        result = close_gaps(result, kernel_size=params["close_gaps_kernel"])
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, f"0{step}_gaps_closed.png"), result)

    if params.get("clahe_enabled", False):
        step += 1
        result = enhance_contrast(
            result,
            clip_limit=params.get("clahe_clip_limit", 2.0),
            grid_size=params.get("clahe_grid_size", 8),
        )
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, f"0{step}_clahe.png"), result)

    if params.get("sharpen_enabled", True):
        step += 1
        result = sharpen(result, amount=params.get("sharpen_amount", 1.5))
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, f"0{step}_sharpened.png"), result)

    return result


# -----------------------------------------------------------------------
# Tile resizing for VLM
# -----------------------------------------------------------------------

def _resize_for_vlm(img: np.ndarray, max_size: int = VLM_TILE_SIZE) -> np.ndarray:
    """Resize image so longest side is max_size, preserving aspect ratio."""
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    scale = max_size / max(h, w)
    new_w = round(w * scale)
    new_h = round(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


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


# -----------------------------------------------------------------------
# CV augmentation
# -----------------------------------------------------------------------

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


def _run_cv_per_tile(tiles: list[Tile]) -> dict:
    """Run CV augmentation per tile."""
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
            f"CV analysis detected {len(tile_lines)} line segments "
            f"and {len(tile_texts)} text labels. "
        )
        if tile_lines:
            horiz = sum(1 for ln in tile_lines if ln["angle"] < 30)
            orient = "horizontal" if horiz > len(tile_lines) / 2 else "mixed"
            ctx += f"Line orientations: mostly {orient}. "
        if tile_texts:
            labels = [t["text"] for t in tile_texts[:10]]
            ctx += f"Detected labels: {', '.join(labels)}. "
        tile_cv_contexts.append(ctx)

    color_info = _cluster_existing_colors(all_lines)

    return {
        "lines_count": total_lines,
        "texts_count": total_texts,
        "shapes_count": total_shapes,
        "color_clusters": len(color_info.get("clusters", [])),
        "tile_cv_contexts": tile_cv_contexts,
        "all_texts": all_texts,
        "all_lines": all_lines,
    }


# -----------------------------------------------------------------------
# VLM analysis
# -----------------------------------------------------------------------

async def _run_vlm(
    global_view: np.ndarray,
    tiles: list[Tile],
    cv_data: dict,
) -> dict:
    """Run global + tile VLM passes (no cross-tile pass)."""
    client = OllamaClient(model=MODEL, base_url=OLLAMA_URL, timeout=TIMEOUT)

    # Pass A: global overview
    print("[net-sight]   VLM: global overview...")
    t = time.time()
    global_result = await run_global_pass(client, global_view)
    print(f"[net-sight]   VLM: global done ({time.time() - t:.0f}s)")

    # Pass B: per-tile detail (sequential with progress)
    tile_results = []
    for i, tile in enumerate(tiles):
        print(f"[net-sight]   VLM: tile {i + 1}/{len(tiles)}...")
        t = time.time()

        # Resize tile for VLM (keep original for CV which already ran)
        vlm_img = _resize_for_vlm(tile.image)
        cv_ctx = cv_data["tile_cv_contexts"][i] if i < len(cv_data["tile_cv_contexts"]) else ""
        prompt = format_prompt(TILE_PROMPT, cv_context=cv_ctx)

        result = await client.analyze_image(vlm_img, prompt)
        tile_results.append(result)
        print(f"[net-sight]   VLM: tile {i + 1}/{len(tiles)} done ({time.time() - t:.0f}s)")

    return {
        "global": global_result,
        "tiles": tile_results,
        "cross_tile": [],
    }
