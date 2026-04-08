"""Prompt templates for the three VLM analysis passes."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Pass A: Global overview
# ---------------------------------------------------------------------------

GLOBAL_PROMPT: str = """\
You are analyzing a network architecture diagram. This is a resized overview of the full image.

Identify and describe:
1. **Overall topology**: star, mesh, hub-and-spoke, hierarchical, hybrid, etc.
2. **Network zones/segments**: DMZ, LAN, WAN, data-center, cloud, management, etc. \
For each zone, state its approximate location in the image (top, bottom, left, right, center).
3. **Legend and abbreviations**: list any legend entries, color codes, line-style meanings, \
or abbreviations visible on the diagram.
4. **Spatial layout**: describe how the diagram is organized (top-to-bottom, left-to-right, \
layered, clustered).

Be precise, use the exact labels you can read. If something is unclear, say so rather than guessing.\
"""

# ---------------------------------------------------------------------------
# Pass B: Tile detail
# ---------------------------------------------------------------------------

TILE_PROMPT: str = """\
You are analyzing a cropped tile from a network architecture diagram.

{cv_context}

List every element you can identify:
1. **Devices/equipment**: routers, switches, firewalls, servers, endpoints, cloud icons, etc. \
Include the exact label text next to each device.
2. **Connections**: for each visible link, state source label, destination label, and link type \
(solid, dashed, color, arrow direction).
3. **Text and labels**: list every readable text string, including IP addresses, VLAN IDs, \
interface names, annotations.

Use the CV data above (if provided) to confirm or supplement your findings. \
Report only what is visible in this tile.\
"""

# ---------------------------------------------------------------------------
# Pass C: Cross-tile connections
# ---------------------------------------------------------------------------

CROSS_TILE_PROMPT: str = """\
You are looking at the overlapping boundary between two adjacent tiles from a network diagram.

{cv_context}

Focus exclusively on connections that cross the boundary:
1. **Border lines**: lines or arrows entering/exiting the visible area at the edges.
2. **Endpoints**: for each border line, identify what it connects to on each side \
(device label, or "exits toward [direction]" if the endpoint is outside the visible area).
3. **Link attributes**: color, line style (solid/dashed), arrowheads.

Ignore elements that are fully contained within a single tile. \
Report only cross-boundary connections.\
"""


def format_prompt(template: str, **kwargs: str) -> str:
    """Substitute placeholders in a prompt template.

    Uses :meth:`str.format_map` with a default-dict so that missing keys
    are left as empty strings rather than raising ``KeyError``.

    Parameters
    ----------
    template:
        A prompt string containing ``{key}`` placeholders.
    **kwargs:
        Values to inject.  Common keys: ``cv_context``.

    Returns
    -------
    str
        The rendered prompt.
    """

    class _Default(dict):
        def __missing__(self, key: str) -> str:
            return ""

    return template.format_map(_Default(**kwargs))
