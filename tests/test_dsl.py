"""Tests for arc/dsl.py.

Each DSL primitive has its own test class.  Tests verify:
  - correct output values / shapes
  - immutability (all DSL functions return new arrays)
  - edge cases (identity transforms, empty regions, etc.)
"""
from __future__ import annotations

import numpy as np
import pytest

from arc.dsl import (
    bounding_box,
    crop,
    crop_to_content,
    find_objects,
    flip,
    flood_fill,
    mask,
    overlay,
    recolor,
    rotate,
    scale,
    tile,
    translate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def g(*rows):
    """Build an int32 Grid from row tuples (convenience shorthand for tests)."""
    return np.array(rows, dtype=np.int32)


# ---------------------------------------------------------------------------
# crop
# ---------------------------------------------------------------------------

class TestCrop:
    def test_basic_crop(self):
        grid = g([0, 1, 2], [3, 4, 5], [6, 7, 8])
        result = crop(grid, 0, 1, 2, 3)
        np.testing.assert_array_equal(result, [[1, 2], [4, 5]])

    def test_full_grid(self):
        grid = g([1, 2], [3, 4])
        np.testing.assert_array_equal(crop(grid, 0, 0, 2, 2), grid)

    def test_single_cell(self):
        grid = g([0, 1], [2, 3])
        np.testing.assert_array_equal(crop(grid, 1, 1, 2, 2), [[3]])

    def test_returns_copy(self):
        """Mutating the result must not affect the source grid."""
        grid = g([1, 2], [3, 4])
        result = crop(grid, 0, 0, 2, 2)
        result[0, 0] = 99
        assert grid[0, 0] == 1


# ---------------------------------------------------------------------------
# rotate
# ---------------------------------------------------------------------------

class TestRotate:
    def test_rotate_90(self):
        grid = g([1, 2], [3, 4])
        expected = g([2, 4], [1, 3])
        np.testing.assert_array_equal(rotate(grid, 1), expected)

    def test_rotate_180(self):
        grid = g([1, 2], [3, 4])
        expected = g([4, 3], [2, 1])
        np.testing.assert_array_equal(rotate(grid, 2), expected)

    def test_rotate_270(self):
        grid = g([1, 2], [3, 4])
        expected = g([3, 1], [4, 2])
        np.testing.assert_array_equal(rotate(grid, 3), expected)

    def test_rotate_360_identity(self):
        """Four 90° rotations should return to the original grid."""
        grid = g([1, 2, 3], [4, 5, 6])
        np.testing.assert_array_equal(rotate(grid, 4), grid)

    def test_returns_copy(self):
        grid = g([1, 2], [3, 4])
        result = rotate(grid, 1)
        result[0, 0] = 99
        assert grid[0, 0] == 1


# ---------------------------------------------------------------------------
# flip
# ---------------------------------------------------------------------------

class TestFlip:
    def test_flip_vertical(self):
        grid = g([1, 2], [3, 4])
        expected = g([3, 4], [1, 2])
        np.testing.assert_array_equal(flip(grid, axis=0), expected)

    def test_flip_horizontal(self):
        grid = g([1, 2], [3, 4])
        expected = g([2, 1], [4, 3])
        np.testing.assert_array_equal(flip(grid, axis=1), expected)

    def test_flip_twice_is_identity(self):
        """Flipping the same axis twice should restore the original."""
        grid = g([1, 2, 3], [4, 5, 6])
        np.testing.assert_array_equal(flip(flip(grid, 0), 0), grid)

    def test_returns_copy(self):
        grid = g([1, 2], [3, 4])
        result = flip(grid, 0)
        result[0, 0] = 99
        assert grid[0, 0] == 1


# ---------------------------------------------------------------------------
# translate
# ---------------------------------------------------------------------------

class TestTranslate:
    def test_shift_down(self):
        grid = g([1, 2], [3, 4])
        result = translate(grid, 1, 0)
        np.testing.assert_array_equal(result, [[0, 0], [1, 2]])

    def test_shift_right(self):
        grid = g([1, 2], [3, 4])
        result = translate(grid, 0, 1)
        np.testing.assert_array_equal(result, [[0, 1], [0, 3]])

    def test_shift_up(self):
        grid = g([1, 2], [3, 4])
        result = translate(grid, -1, 0)
        np.testing.assert_array_equal(result, [[3, 4], [0, 0]])

    def test_shift_left(self):
        grid = g([1, 2], [3, 4])
        result = translate(grid, 0, -1)
        np.testing.assert_array_equal(result, [[2, 0], [4, 0]])

    def test_zero_shift_identity(self):
        grid = g([1, 2], [3, 4])
        np.testing.assert_array_equal(translate(grid, 0, 0), grid)

    def test_custom_fill(self):
        grid = g([1, 2], [3, 4])
        result = translate(grid, 1, 0, fill=9)
        assert result[0, 0] == 9

    def test_shift_fully_off(self):
        """Shifting by more than the grid size should yield an all-fill grid."""
        grid = g([1, 2], [3, 4])
        result = translate(grid, 5, 0)
        np.testing.assert_array_equal(result, np.zeros_like(grid))


# ---------------------------------------------------------------------------
# scale
# ---------------------------------------------------------------------------

class TestScale:
    def test_scale_2x(self):
        grid = g([1, 2], [3, 4])
        expected = g([1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4])
        np.testing.assert_array_equal(scale(grid, 2), expected)

    def test_scale_1x_identity(self):
        grid = g([1, 2], [3, 4])
        np.testing.assert_array_equal(scale(grid, 1), grid)

    def test_scale_output_shape(self):
        grid = g([1, 2, 3], [4, 5, 6])
        result = scale(grid, 3)
        assert result.shape == (6, 9)


# ---------------------------------------------------------------------------
# tile
# ---------------------------------------------------------------------------

class TestTile:
    def test_tile_2x2(self):
        grid = g([1, 0], [0, 1])
        result = tile(grid, 2, 2)
        assert result.shape == (4, 4)
        np.testing.assert_array_equal(result[:2, :2], grid)
        np.testing.assert_array_equal(result[:2, 2:], grid)

    def test_tile_1x1_identity(self):
        grid = g([1, 2], [3, 4])
        np.testing.assert_array_equal(tile(grid, 1, 1), grid)

    def test_tile_output_shape(self):
        grid = g([1, 2], [3, 4])
        assert tile(grid, 3, 4).shape == (6, 8)


# ---------------------------------------------------------------------------
# recolor
# ---------------------------------------------------------------------------

class TestRecolor:
    def test_basic_recolor(self):
        grid = g([0, 1, 0], [1, 0, 1])
        result = recolor(grid, 1, 2)
        expected = g([0, 2, 0], [2, 0, 2])
        np.testing.assert_array_equal(result, expected)

    def test_noop_when_color_absent(self):
        """Recoloring a colour not present in the grid should leave it unchanged."""
        grid = g([0, 1], [1, 0])
        np.testing.assert_array_equal(recolor(grid, 5, 9), grid)

    def test_returns_copy(self):
        grid = g([1, 2], [3, 4])
        result = recolor(grid, 1, 9)
        assert grid[0, 0] == 1  # original unchanged

    def test_recolor_all(self):
        grid = np.ones((3, 3), dtype=np.int32)
        result = recolor(grid, 1, 7)
        assert np.all(result == 7)


# ---------------------------------------------------------------------------
# mask
# ---------------------------------------------------------------------------

class TestMask:
    def test_mask_zeros_out(self):
        grid = g([1, 2], [3, 4])
        mask_grid = g([1, 0], [0, 1])
        result = mask(grid, mask_grid)
        np.testing.assert_array_equal(result, [[1, 0], [0, 4]])

    def test_mask_all_ones_identity(self):
        """A mask of all 1s should leave the grid unchanged."""
        grid = g([1, 2], [3, 4])
        mask_grid = np.ones_like(grid)
        np.testing.assert_array_equal(mask(grid, mask_grid), grid)

    def test_mask_custom_fill(self):
        grid = g([1, 2], [3, 4])
        mask_grid = g([0, 1], [1, 0])
        result = mask(grid, mask_grid, fill=9)
        assert result[0, 0] == 9
        assert result[1, 1] == 9


# ---------------------------------------------------------------------------
# overlay
# ---------------------------------------------------------------------------

class TestOverlay:
    def test_overlay_non_transparent(self):
        """Non-transparent cells in top should overwrite base."""
        base = g([0, 0], [0, 0])
        top  = g([1, 0], [0, 2])
        result = overlay(base, top, transparent=0)
        np.testing.assert_array_equal(result, [[1, 0], [0, 2]])

    def test_transparent_cells_not_painted(self):
        """Cells in top equal to transparent_value should leave base unchanged."""
        base = g([5, 5], [5, 5])
        top  = g([0, 3], [0, 0])
        result = overlay(base, top, transparent=0)
        np.testing.assert_array_equal(result, [[5, 3], [5, 5]])

    def test_full_overlap(self):
        """When transparent_value cannot appear in top, all cells are overwritten."""
        base = g([1, 1], [1, 1])
        top  = g([2, 2], [2, 2])
        np.testing.assert_array_equal(overlay(base, top, transparent=-1), top)


# ---------------------------------------------------------------------------
# flood_fill
# ---------------------------------------------------------------------------

class TestFloodFill:
    def test_basic_fill(self):
        """All-same-colour grid should be fully filled."""
        grid = g([0, 0, 0], [0, 0, 0], [0, 0, 0])
        result = flood_fill(grid, 0, 0, 5)
        assert np.all(result == 5)

    def test_bounded_fill(self):
        """Fill should be blocked by cells of a different colour."""
        grid = g([1, 1, 1], [1, 0, 1], [1, 1, 1])
        result = flood_fill(grid, 1, 1, 3)
        assert result[1, 1] == 3
        assert result[0, 0] == 1  # border unchanged

    def test_same_color_noop(self):
        """Filling with the same colour as the target is a no-op."""
        grid = g([1, 1], [1, 1])
        result = flood_fill(grid, 0, 0, 1)
        np.testing.assert_array_equal(result, grid)

    def test_does_not_cross_border(self):
        """4-connectivity: diagonal corners must not be reached."""
        grid = g([1, 0, 1], [0, 0, 0], [1, 0, 1])
        result = flood_fill(grid, 1, 1, 9)
        assert result[0, 0] == 1  # isolated by 0s
        assert result[1, 1] == 9

    def test_returns_copy(self):
        grid = g([0, 0], [0, 0])
        result = flood_fill(grid, 0, 0, 5)
        assert grid[0, 0] == 0  # original unchanged


# ---------------------------------------------------------------------------
# find_objects
# ---------------------------------------------------------------------------

class TestFindObjects:
    def test_single_object(self):
        grid = g([0, 0, 0], [0, 1, 0], [0, 0, 0])
        objs = find_objects(grid, background=0)
        assert len(objs) == 1
        assert objs[0]["color"] == 1

    def test_two_separate_objects(self):
        grid = g([1, 0, 2], [0, 0, 0], [0, 0, 0])
        objs = find_objects(grid, background=0)
        colors = {o["color"] for o in objs}
        assert colors == {1, 2}

    def test_connected_object_pixels(self):
        """Three horizontally-connected cells should form one object."""
        grid = g([1, 1, 0], [1, 0, 0])
        objs = find_objects(grid, background=0)
        assert len(objs) == 1
        assert len(objs[0]["pixels"]) == 3

    def test_bbox_correct(self):
        grid = g([0, 0, 0], [0, 1, 0], [0, 1, 0])
        objs = find_objects(grid, background=0)
        assert len(objs) == 1
        r_min, c_min, r_max, c_max = objs[0]["bbox"]
        assert r_min == 1 and r_max == 2
        assert c_min == 1 and c_max == 1

    def test_subgrid_shape(self):
        """Subgrid should be as small as the object's bounding box."""
        grid = g([0, 0, 0], [0, 1, 1], [0, 0, 0])
        objs = find_objects(grid, background=0)
        assert objs[0]["subgrid"].shape == (1, 2)

    def test_empty_grid_returns_nothing(self):
        """A grid that is entirely background has no objects."""
        grid = np.zeros((3, 3), dtype=np.int32)
        objs = find_objects(grid, background=0)
        assert objs == []

    def test_infers_background(self):
        """When background is not specified, the most frequent colour is used."""
        grid = g([0, 0, 0], [0, 1, 0], [0, 0, 0])
        objs = find_objects(grid)  # background inferred as 0
        assert len(objs) == 1

    def test_multicolor_each_separate(self):
        """Each distinct non-background colour in its own cell is its own object."""
        grid = g([1, 0, 2, 0, 3])
        objs = find_objects(grid, background=0)
        assert len(objs) == 3


# ---------------------------------------------------------------------------
# bounding_box
# ---------------------------------------------------------------------------

class TestBoundingBox:
    def test_full_grid(self):
        grid = g([1, 1], [1, 1])
        assert bounding_box(grid, color=1) == (0, 0, 1, 1)

    def test_single_cell(self):
        grid = g([0, 0, 0], [0, 1, 0], [0, 0, 0])
        assert bounding_box(grid, color=1) == (1, 1, 1, 1)

    def test_non_background_bbox(self):
        """Without a color argument, bounding_box covers all non-background cells."""
        grid = g([0, 0, 0], [0, 1, 1], [0, 1, 0])
        r_min, c_min, r_max, c_max = bounding_box(grid)
        assert r_min == 1
        assert c_min == 1


# ---------------------------------------------------------------------------
# crop_to_content
# ---------------------------------------------------------------------------

class TestCropToContent:
    def test_strips_padding(self):
        """A single non-background cell surrounded by zeros → 1×1 output."""
        grid = g([0, 0, 0], [0, 1, 0], [0, 0, 0])
        result = crop_to_content(grid)
        assert result.shape == (1, 1)
        assert result[0, 0] == 1

    def test_no_padding(self):
        """Content flush against all edges should preserve the full extent."""
        grid = g([0, 1, 0], [0, 1, 0])
        result = crop_to_content(grid)
        assert result.shape == (2, 1)
        assert np.all(result == 1)

    def test_rectangular_content(self):
        grid = g([0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0])
        result = crop_to_content(grid)
        assert result.shape == (2, 2)
