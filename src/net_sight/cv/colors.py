"""Colour analysis for detected lines: clustering and legend matching.

K-means is implemented with pure numpy to avoid pulling in scikit-learn.
"""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Lightweight K-means (numpy only)
# ---------------------------------------------------------------------------

def _kmeans(
    data: np.ndarray,
    k: int,
    max_iter: int = 50,
    n_init: int = 10,
    rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run K-means clustering on *data* (N, D).

    Returns (labels, centers, inertia) for the best of *n_init* runs
    (lowest inertia).
    """
    rng = np.random.default_rng(rng_seed)
    n = len(data)
    best_inertia = float("inf")
    best_labels = np.zeros(n, dtype=int)
    best_centers = data[:k].copy()

    for _ in range(n_init):
        # K-means++ initialisation
        idx = [rng.integers(0, n)]
        for _ in range(1, k):
            dists = np.min(
                np.linalg.norm(data[:, None] - data[np.array(idx)][None, :], axis=2),
                axis=1,
            )
            probs = dists ** 2
            total = probs.sum()
            if total == 0:
                idx.append(rng.integers(0, n))
            else:
                probs /= total
                idx.append(int(rng.choice(n, p=probs)))

        centers = data[np.array(idx)].astype(float)

        for _ in range(max_iter):
            # Assign
            dists = np.linalg.norm(data[:, None] - centers[None, :], axis=2)  # (N, k)
            labels = np.argmin(dists, axis=1)
            # Update
            new_centers = np.empty_like(centers)
            for ci in range(k):
                members = data[labels == ci]
                if len(members) == 0:
                    new_centers[ci] = data[rng.integers(0, n)]
                else:
                    new_centers[ci] = members.mean(axis=0)
            if np.allclose(new_centers, centers, atol=1e-4):
                centers = new_centers
                break
            centers = new_centers

        # Inertia
        dists = np.linalg.norm(data - centers[labels], axis=1)
        inertia = float(np.sum(dists ** 2))

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    return best_labels, best_centers, best_inertia


def _auto_k(data: np.ndarray, max_k: int = 12) -> int:
    """Choose optimal *k* for K-means via the elbow method.

    Computes inertia for k = 2 .. max_k and stops when the relative
    reduction drops below 20 %.
    """
    if len(data) < 2:
        return 1
    max_k = min(max_k, len(data))
    if max_k < 2:
        return 1

    prev_inertia: float | None = None
    best_k = 2
    for k in range(2, max_k + 1):
        _, _, inertia = _kmeans(data, k, n_init=3)
        if prev_inertia is not None:
            ratio = inertia / prev_inertia if prev_inertia > 0 else 1.0
            if ratio > 0.8:
                break
        prev_inertia = inertia
        best_k = k

    return best_k


# ---------------------------------------------------------------------------
# Colour sampling
# ---------------------------------------------------------------------------

def _sample_line_colors(
    img: np.ndarray, lines: list[dict], n_per_line: int = 10
) -> np.ndarray:
    """Collect RGB samples along every line segment.

    Returns an (N, 3) uint8 array of RGB values.
    """
    if img.ndim == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img

    h, w = img_bgr.shape[:2]
    samples: list[np.ndarray] = []

    for ln in lines:
        xs = np.linspace(ln["x1"], ln["x2"], n_per_line).astype(int)
        ys = np.linspace(ln["y1"], ln["y2"], n_per_line).astype(int)
        xs = np.clip(xs, 0, w - 1)
        ys = np.clip(ys, 0, h - 1)
        bgr = img_bgr[ys, xs]
        rgb = bgr[:, ::-1]
        samples.append(rgb)

    if not samples:
        return np.empty((0, 3), dtype=np.uint8)

    return np.concatenate(samples, axis=0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_line_colors(
    img: np.ndarray, lines: list[dict]
) -> list[tuple[int, int, int]]:
    """Sample the mean RGB colour for each line segment.

    Returns one ``(r, g, b)`` tuple per line in *lines*.
    """
    if img.ndim == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img

    h, w = img_bgr.shape[:2]
    colors: list[tuple[int, int, int]] = []

    for ln in lines:
        xs = np.linspace(ln["x1"], ln["x2"], 20).astype(int)
        ys = np.linspace(ln["y1"], ln["y2"], 20).astype(int)
        xs = np.clip(xs, 0, w - 1)
        ys = np.clip(ys, 0, h - 1)
        bgr = img_bgr[ys, xs].mean(axis=0).astype(int)
        colors.append((int(bgr[2]), int(bgr[1]), int(bgr[0])))

    return colors


def cluster_colors(
    colors: list[tuple[int, int, int]],
    n_clusters: int | None = None,
) -> dict[str, tuple[int, int, int]]:
    """Cluster a list of RGB colours with K-means.

    When *n_clusters* is ``None`` the optimal number is auto-detected via
    the elbow method.

    Returns a dict mapping a cluster label (``"cluster_0"``, ...) to its
    centre ``(r, g, b)``.
    """
    if not colors:
        return {}

    data = np.array(colors, dtype=float)
    # Deduplicate near-identical rows for efficiency
    if len(data) < 2:
        return {"cluster_0": tuple(int(v) for v in data[0])}

    if n_clusters is None:
        k = _auto_k(data)
    else:
        k = min(n_clusters, len(data))

    labels, centers, _ = _kmeans(data, k)

    result: dict[str, tuple[int, int, int]] = {}
    for i in range(k):
        c = centers[i].astype(int)
        result[f"cluster_{i}"] = (int(c[0]), int(c[1]), int(c[2]))

    return result


def build_color_legend(
    clusters: dict[str, tuple[int, int, int]]
) -> dict[str, str]:
    """Map each cluster to a human-readable descriptive label.

    Heuristic: assign a common colour name based on the dominant RGB
    channel.  For example ``(0, 50, 200)`` becomes ``"blue_lines"``.
    """

    def _name_rgb(r: int, g: int, b: int) -> str:
        mx = max(r, g, b)
        if mx < 40:
            return "black_lines"
        mn = min(r, g, b)
        if mx - mn < 30:
            if mx > 180:
                return "white_lines"
            return "gray_lines"
        if r >= g and r >= b:
            if g > b and g > 100:
                return "yellow_lines"
            return "red_lines"
        if g >= r and g >= b:
            if b > r and b > 100:
                return "teal_lines"
            return "green_lines"
        # b is dominant
        if r > g and r > 100:
            return "purple_lines"
        return "blue_lines"

    legend: dict[str, str] = {}
    seen_names: dict[str, int] = {}
    for key, (r, g, b) in clusters.items():
        name = _name_rgb(r, g, b)
        count = seen_names.get(name, 0)
        if count > 0:
            legend[key] = f"{name}_{count}"
        else:
            legend[key] = name
        seen_names[name] = count + 1

    return legend


# ---------------------------------------------------------------------------
# Higher-level helpers (used by pipeline and legacy callers)
# ---------------------------------------------------------------------------

def analyze_line_colors(
    img: np.ndarray, lines: list[dict], n_clusters: int = 8
) -> dict:
    """Cluster the colours sampled along detected lines with K-means.

    Returns::

        {
            "clusters": [
                {"color_rgb": (r, g, b), "count": int, "percentage": float},
                ...
            ],
            "total_lines": int,
        }
    """
    all_samples = _sample_line_colors(img, lines)

    if len(all_samples) == 0:
        return {"clusters": [], "total_lines": 0}

    k = min(n_clusters, len(all_samples))
    labels, centers, _ = _kmeans(all_samples.astype(float), k)

    total = len(labels)
    clusters: list[dict] = []
    for i in range(k):
        count = int(np.sum(labels == i))
        c = centers[i].astype(int)
        clusters.append({
            "color_rgb": (int(c[0]), int(c[1]), int(c[2])),
            "count": count,
            "percentage": round(count / total * 100, 2),
        })

    clusters.sort(key=lambda c: c["count"], reverse=True)

    return {"clusters": clusters, "total_lines": len(lines)}


def match_legend_colors(
    legend_colors: list[tuple], line_clusters: list[dict]
) -> dict:
    """Map detected line-colour clusters to legend colours.

    *legend_colors* is a list of ``(label, (r, g, b))`` pairs coming from
    the legend extraction step.

    *line_clusters* is the ``"clusters"`` list returned by
    :func:`analyze_line_colors`.

    Returns a dict mapping each cluster index to the closest legend label,
    using Euclidean distance in RGB space::

        {0: "VLAN-10", 1: "Management", ...}
    """
    if not legend_colors or not line_clusters:
        return {}

    legend_arr = np.array([c[1] for c in legend_colors], dtype=float)
    labels = [c[0] for c in legend_colors]

    mapping: dict[int, str] = {}
    for idx, cluster in enumerate(line_clusters):
        c_rgb = np.array(cluster["color_rgb"], dtype=float)
        distances = np.linalg.norm(legend_arr - c_rgb, axis=1)
        best = int(np.argmin(distances))
        mapping[idx] = labels[best]

    return mapping
