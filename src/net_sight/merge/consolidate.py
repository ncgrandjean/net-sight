"""Merge and deduplicate results from multi-pass VLM analysis."""

from __future__ import annotations


def merge_tile_results(
    global_result: str,
    tile_results: list[dict],
    cross_tile_results: list[str],
) -> str:
    """Consolidate all VLM pass results into a single structured description.

    Parameters
    ----------
    global_result : str
        Output from the global overview pass (pass A).
    tile_results : list[dict]
        Each dict has keys: "row", "col", "x", "y", "result" (the VLM output text).
    cross_tile_results : list[str]
        Outputs from cross-tile connection passes (pass C).

    Returns
    -------
    str
        Unified markdown description of the entire diagram.
    """
    sections = []

    # --- Section 1: Global overview ---
    sections.append("## 1. OVERVIEW\n")
    sections.append(global_result.strip())
    sections.append("")

    # --- Section 2: Detailed analysis per zone ---
    sections.append("## 2. DETAILED ANALYSIS BY ZONE\n")
    for tile in sorted(tile_results, key=lambda t: (t["row"], t["col"])):
        label = f"Zone (row {tile['row']}, col {tile['col']}) at ({tile['x']}, {tile['y']})"
        sections.append(f"### {label}\n")
        sections.append(tile["result"].strip())
        sections.append("")

    # --- Section 3: Cross-tile connections ---
    if cross_tile_results:
        sections.append("## 3. CROSS-ZONE CONNECTIONS\n")
        for i, result in enumerate(cross_tile_results, 1):
            text = result.strip()
            if text:
                sections.append(f"### Connection group {i}\n")
                sections.append(text)
                sections.append("")

    # --- Section 4: Consolidated inventory ---
    sections.append("## 4. CONSOLIDATED INVENTORY\n")
    sections.append(_build_inventory(global_result, tile_results, cross_tile_results))

    return "\n".join(sections)


def _build_inventory(
    global_result: str,
    tile_results: list[dict],
    cross_tile_results: list[str],
) -> str:
    """Build a summary inventory section from all results.

    This is a best-effort deduplication by collecting all bullet-point items
    that look like device/connection entries across all passes.
    """
    all_text = global_result + "\n"
    for t in tile_results:
        all_text += t["result"] + "\n"
    for c in cross_tile_results:
        all_text += c + "\n"

    # Extract unique lines that look like inventory items (start with - or *)
    seen = set()
    items = []
    for line in all_text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("-", "*")) and len(stripped) > 3:
            # Normalize for dedup: lowercase, strip bullets and whitespace
            key = stripped.lstrip("-* ").lower().strip()
            if key not in seen:
                seen.add(key)
                items.append(stripped)

    if items:
        return "\n".join(items)
    return "No structured inventory items detected."
