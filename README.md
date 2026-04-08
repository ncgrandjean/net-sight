# net-sight

Network architecture diagram analyzer. Combines OpenCV preprocessing, intelligent tiling, and multi-pass VLM analysis (Ollama/Qwen3-VL) to produce detailed markdown descriptions of network diagrams.

## Features

- **Auto-calibrated preprocessing**: line thickness detection, morphological enhancement, contrast optimization
- **Intelligent tiling**: auto-sized grid with overlap for large/dense diagrams
- **CV augmentation**: line detection (Hough), OCR (EasyOCR), shape detection, color clustering
- **Multi-pass VLM analysis**: global overview + per-tile detail + cross-tile connection stitching
- **Parallel processing**: async Ollama requests with configurable worker count

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com/) running locally with `qwen3-vl:8b` (or another VL model)

## Installation

```bash
pip install -e .
```

## Usage

```bash
net-sight schema.png
```

Produces `schema.md` in the same directory.

## Configuration

Edit the variables at the top of `src/net_sight/pipeline.py`:

```python
WORKERS = 4              # Parallel VLM requests
MODEL = "qwen3-vl:8b"   # Ollama model name
OLLAMA_URL = "http://localhost:11434"
TIMEOUT = 300            # Per-request timeout (seconds)
```

## Pipeline

```
PNG -> [Preprocess] -> [Tile] -> [CV Analysis] -> [VLM Multi-Pass] -> [Merge] -> MD
```

1. **Preprocess**: auto-detects line thickness, upscales if needed, dilates thin lines, enhances contrast
2. **Tile**: splits image into overlapping tiles sized for the VLM (1024px target)
3. **CV Analysis**: extracts lines, text (OCR), shapes, and color categories
4. **VLM Analysis**: 3 parallel passes (global topology, per-tile detail, cross-tile connections)
5. **Merge**: consolidates and deduplicates results into structured markdown
