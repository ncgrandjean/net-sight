"""Orchestration of the three VLM analysis passes (global, tile, cross-tile)."""

from __future__ import annotations

import logging

import numpy as np

from net_sight.analyze.ollama_client import OllamaClient
from net_sight.analyze.prompts import (
    CROSS_TILE_PROMPT,
    GLOBAL_PROMPT,
    TILE_PROMPT,
    format_prompt,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Pass A: Global overview
# ------------------------------------------------------------------


async def run_global_pass(
    client: OllamaClient,
    global_view: np.ndarray,
    cv_context: str = "",
) -> str:
    """Run pass A on the resized overview image.

    Parameters
    ----------
    client:
        Configured :class:`OllamaClient`.
    global_view:
        Resized image (longest side ~ 1024 px).
    cv_context:
        Optional CV-extracted context to append to the prompt.

    Returns
    -------
    str
        Raw VLM response describing the global topology.
    """
    logger.info("Pass A (global): analyzing overview image")
    prompt = GLOBAL_PROMPT
    if cv_context:
        prompt += f"\n\nCV pre-analysis data:\n{cv_context}"
    result = await client.analyze_image(global_view, prompt)
    logger.info("Pass A (global): done")
    return result


# ------------------------------------------------------------------
# Pass B: Tile detail (parallel)
# ------------------------------------------------------------------


async def run_tile_pass(
    client: OllamaClient,
    tiles: list[tuple[dict, np.ndarray]],
    cv_contexts: list[str],
    workers: int = 4,
) -> list[str]:
    """Run pass B on every tile in parallel.

    Parameters
    ----------
    client:
        Configured :class:`OllamaClient`.
    tiles:
        List of ``(metadata, tile_image)`` as returned by
        :func:`~net_sight.tiling.grid.extract_tiles`.
    cv_contexts:
        One CV context string per tile (same order). Pass empty strings
        if no CV data is available.
    workers:
        Maximum number of concurrent VLM requests.

    Returns
    -------
    list[str]
        One VLM response per tile, in the same order as *tiles*.
    """
    total = len(tiles)
    logger.info("Pass B (tile detail): %d tiles, %d workers", total, workers)

    tasks: list[tuple[np.ndarray, str]] = []
    for i, ((meta, tile_img), cv_ctx) in enumerate(zip(tiles, cv_contexts)):
        label = f"tile ({meta['row']},{meta['col']})"
        logger.info("Pass B: preparing %s (%d/%d)", label, i + 1, total)
        prompt = format_prompt(TILE_PROMPT, cv_context=cv_ctx)
        tasks.append((tile_img, prompt))

    results = await client.analyze_image_batch(tasks, workers=workers)
    logger.info("Pass B (tile detail): done")
    return results


# ------------------------------------------------------------------
# Pass C: Cross-tile connections (parallel)
# ------------------------------------------------------------------


async def run_cross_tile_pass(
    client: OllamaClient,
    adjacent_pairs: list[tuple[dict, dict]],
    pair_images: list[np.ndarray],
    cv_contexts: list[str],
    workers: int = 4,
) -> list[str]:
    """Run pass C on overlap regions between adjacent tiles.

    Parameters
    ----------
    client:
        Configured :class:`OllamaClient`.
    adjacent_pairs:
        List of ``(tile_meta_a, tile_meta_b)`` for adjacent tiles.
    pair_images:
        Pre-cropped images covering the overlap region of each pair.
    cv_contexts:
        One CV context string per pair (same order).
    workers:
        Maximum number of concurrent VLM requests.

    Returns
    -------
    list[str]
        One VLM response per adjacent pair, in the same order.
    """
    total = len(adjacent_pairs)
    logger.info("Pass C (cross-tile): %d pairs, %d workers", total, workers)

    tasks: list[tuple[np.ndarray, str]] = []
    for i, ((meta_a, meta_b), pair_img, cv_ctx) in enumerate(
        zip(adjacent_pairs, pair_images, cv_contexts)
    ):
        label = f"({meta_a['row']},{meta_a['col']})-({meta_b['row']},{meta_b['col']})"
        logger.info("Pass C: preparing pair %s (%d/%d)", label, i + 1, total)
        prompt = format_prompt(CROSS_TILE_PROMPT, cv_context=cv_ctx)
        tasks.append((pair_img, prompt))

    results = await client.analyze_image_batch(tasks, workers=workers)
    logger.info("Pass C (cross-tile): done")
    return results


# ------------------------------------------------------------------
# Full pipeline
# ------------------------------------------------------------------


async def run_all_passes(
    client: OllamaClient,
    global_view: np.ndarray,
    tiles: list[tuple[dict, np.ndarray]],
    adjacent_pairs: list[tuple[dict, dict]],
    pair_images: list[np.ndarray],
    cv_context_global: str = "",
    cv_contexts_tiles: list[str] | None = None,
    cv_contexts_pairs: list[str] | None = None,
    workers: int = 4,
) -> dict:
    """Orchestrate passes A, B, and C and return consolidated results.

    Parameters
    ----------
    client:
        Configured :class:`OllamaClient`.
    global_view:
        Resized overview image for pass A.
    tiles:
        ``(metadata, tile_image)`` pairs for pass B.
    adjacent_pairs:
        Adjacent tile metadata pairs for pass C.
    pair_images:
        Overlap-region images for pass C (one per pair).
    cv_context_global:
        CV context for the global pass.
    cv_contexts_tiles:
        CV context per tile. Defaults to empty strings.
    cv_contexts_pairs:
        CV context per adjacent pair. Defaults to empty strings.
    workers:
        Maximum concurrent VLM requests for passes B and C.

    Returns
    -------
    dict
        ``{"global": str, "tiles": list[str], "cross_tile": list[str]}``.
    """
    if cv_contexts_tiles is None:
        cv_contexts_tiles = [""] * len(tiles)
    if cv_contexts_pairs is None:
        cv_contexts_pairs = [""] * len(adjacent_pairs)

    # Pass A: global overview (single request)
    global_result = await run_global_pass(client, global_view, cv_context_global)

    # Pass B: tile details (parallel batch)
    tile_results = await run_tile_pass(client, tiles, cv_contexts_tiles, workers)

    # Pass C: cross-tile connections (parallel batch)
    cross_results = await run_cross_tile_pass(
        client, adjacent_pairs, pair_images, cv_contexts_pairs, workers
    )

    return {
        "global": global_result,
        "tiles": tile_results,
        "cross_tile": cross_results,
    }
