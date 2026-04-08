"""VLM multi-pass analysis module for network architecture diagrams."""

from net_sight.analyze.ollama_client import OllamaClient
from net_sight.analyze.passes import (
    run_all_passes,
    run_cross_tile_pass,
    run_global_pass,
    run_tile_pass,
)
from net_sight.analyze.prompts import (
    CROSS_TILE_PROMPT,
    GLOBAL_PROMPT,
    TILE_PROMPT,
    format_prompt,
)

__all__ = [
    "CROSS_TILE_PROMPT",
    "GLOBAL_PROMPT",
    "OllamaClient",
    "TILE_PROMPT",
    "format_prompt",
    "run_all_passes",
    "run_cross_tile_pass",
    "run_global_pass",
    "run_tile_pass",
]
