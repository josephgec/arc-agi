"""Shared fixtures for ARC-AGI tests."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def simple_task():
    """A minimal hand-crafted task (recolor: 1→2)."""
    return {
        "train": [
            {
                "input": np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int32),
                "output": np.array([[0, 2, 0], [2, 0, 2]], dtype=np.int32),
            },
            {
                "input": np.array([[1, 1, 0]], dtype=np.int32),
                "output": np.array([[2, 2, 0]], dtype=np.int32),
            },
        ],
        "test": [
            {"input": np.array([[0, 1, 1]], dtype=np.int32)},
        ],
    }


@pytest.fixture
def tile_task():
    """A task where the output is the input tiled 2×2."""
    inp = np.array([[1, 0], [0, 1]], dtype=np.int32)
    out = np.tile(inp, (2, 2))
    return {
        "train": [{"input": inp, "output": out}],
        "test": [{"input": inp}],
    }


@pytest.fixture
def task_json_file(tmp_path):
    """Write a simple task JSON to a temp file and return its path."""
    raw = {
        "train": [
            {"input": [[0, 1], [1, 0]], "output": [[0, 2], [2, 0]]},
            {"input": [[1, 0], [0, 1]], "output": [[2, 0], [0, 2]]},
        ],
        "test": [
            {"input": [[1, 1], [0, 0]], "output": [[2, 2], [0, 0]]},
        ],
    }
    p = tmp_path / "test_task.json"
    p.write_text(json.dumps(raw))
    return p


@pytest.fixture
def training_dir():
    """Return path to the real ARC training data (skip if not present)."""
    p = Path(__file__).parent.parent / "data" / "data" / "training"
    if not p.exists():
        pytest.skip("ARC training data not found")
    return p
