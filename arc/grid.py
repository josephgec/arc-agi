"""Core grid utilities for ARC-AGI tasks."""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Any

# ARC colour palette (index → name)
COLOR_NAMES = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "grey",
    6: "magenta",
    7: "orange",
    8: "azure",
    9: "maroon",
}

Grid = np.ndarray  # 2-D integer array, dtype=int32


def load_task(path: str | Path) -> dict[str, Any]:
    """Load an ARC task JSON file.

    Returns a dict with keys:
        'train': list of {'input': Grid, 'output': Grid}
        'test':  list of {'input': Grid, 'output': Grid | None}
    """
    with open(path) as f:
        raw = json.load(f)

    def to_grid(lst: list[list[int]]) -> Grid:
        return np.array(lst, dtype=np.int32)

    task: dict[str, Any] = {"train": [], "test": []}
    for pair in raw.get("train", []):
        task["train"].append(
            {"input": to_grid(pair["input"]), "output": to_grid(pair["output"])}
        )
    for pair in raw.get("test", []):
        entry: dict[str, Any] = {"input": to_grid(pair["input"])}
        if "output" in pair:
            entry["output"] = to_grid(pair["output"])
        task["test"].append(entry)

    return task


def grids_equal(a: Grid, b: Grid) -> bool:
    """Return True if two grids are identical in shape and values."""
    return a.shape == b.shape and np.array_equal(a, b)


def grid_from_list(lst: list[list[int]]) -> Grid:
    return np.array(lst, dtype=np.int32)


def grid_to_list(grid: Grid) -> list[list[int]]:
    return grid.tolist()


def unique_colors(grid: Grid) -> list[int]:
    return sorted(int(c) for c in np.unique(grid))


def background_color(grid: Grid) -> int:
    """Return the most-frequent colour (assumed background)."""
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])
