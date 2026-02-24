"""Shared pytest fixtures for ARC-AGI tests."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def simple_task():
    """A minimal hand-crafted task whose rule is: recolor 1 → 2.

    Two training pairs so tests that check multi-pair behaviour have enough data.
    """
    return {
        "train": [
            {
                "input":  np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int32),
                "output": np.array([[0, 2, 0], [2, 0, 2]], dtype=np.int32),
            },
            {
                "input":  np.array([[1, 1, 0]], dtype=np.int32),
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
        "test":  [{"input": inp}],
    }


@pytest.fixture
def task_json_file(tmp_path):
    """Write a simple recolor task to a temp JSON file and return its Path.

    The file is named 'test_task.json' so that task_id == 'test_task'.
    """
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
    """Return the path to real ARC training data, skipping the test if absent."""
    p = Path(__file__).parent.parent / "data" / "data" / "training"
    if not p.exists():
        pytest.skip("ARC training data not found")
    return p
