"""DSL of common ARC grid transformation primitives.

Every function takes and returns numpy int32 arrays (Grid).
All operations are pure — they return a new array and never mutate the input.
"""
from __future__ import annotations

import numpy as np
from .grid import Grid, background_color


# ---------------------------------------------------------------------------
# Geometric transforms
# ---------------------------------------------------------------------------

def crop(grid: Grid, r1: int, c1: int, r2: int, c2: int) -> Grid:
    """Return the sub-grid at rows [r1:r2], cols [c1:c2] (exclusive end)."""
    return grid[r1:r2, c1:c2].copy()


def rotate(grid: Grid, n: int = 1) -> Grid:
    """Rotate the grid 90° counter-clockwise n times."""
    return np.rot90(grid, n).copy()


def flip(grid: Grid, axis: int = 0) -> Grid:
    """Flip the grid along an axis: 0 = vertical (up/down), 1 = horizontal (left/right)."""
    return np.flip(grid, axis=axis).copy()


def translate(grid: Grid, dr: int, dc: int, fill: int = 0) -> Grid:
    """Shift the grid by (dr rows, dc cols), filling vacated cells with `fill`.

    Positive dr shifts down; positive dc shifts right.
    Cells that scroll off an edge are discarded (no wrapping).
    """
    result = np.full_like(grid, fill)
    rows, cols = grid.shape

    # Compute the source and destination row ranges for the copy.
    # If dr > 0 we shift down: source starts at row 0, destination at row dr.
    # If dr < 0 we shift up:   source starts at row -dr, destination at row 0.
    src_r = max(0, -dr), max(0, min(rows, rows - dr))
    dst_r = max(0,  dr), max(0, min(rows, rows + dr))

    # Same logic for columns.
    src_c = max(0, -dc), max(0, min(cols, cols - dc))
    dst_c = max(0,  dc), max(0, min(cols, cols + dc))

    result[dst_r[0]:dst_r[1], dst_c[0]:dst_c[1]] = grid[src_r[0]:src_r[1], src_c[0]:src_c[1]]
    return result


def scale(grid: Grid, factor: int) -> Grid:
    """Scale up a grid by an integer factor (each cell becomes a factor×factor block)."""
    return np.kron(grid, np.ones((factor, factor), dtype=np.int32))


def tile(grid: Grid, n_rows: int, n_cols: int) -> Grid:
    """Repeat the grid n_rows times vertically and n_cols times horizontally."""
    return np.tile(grid, (n_rows, n_cols))


# ---------------------------------------------------------------------------
# Colour operations
# ---------------------------------------------------------------------------

def recolor(grid: Grid, from_color: int, to_color: int) -> Grid:
    """Replace every occurrence of from_color with to_color."""
    result = grid.copy()
    result[result == from_color] = to_color
    return result


def mask(grid: Grid, mask_grid: Grid, fill: int = 0) -> Grid:
    """Zero out (fill) cells where mask_grid == 0, keeping all other cells."""
    result = grid.copy()
    result[mask_grid == 0] = fill
    return result


def overlay(base: Grid, top: Grid, transparent: int = 0) -> Grid:
    """Overlay `top` onto `base`, skipping cells where top == transparent."""
    result = base.copy()
    result[top != transparent] = top[top != transparent]
    return result


# ---------------------------------------------------------------------------
# Flood fill
# ---------------------------------------------------------------------------

def flood_fill(grid: Grid, row: int, col: int, new_color: int) -> Grid:
    """Flood-fill starting at (row, col), replacing all connected same-color cells.

    Uses 4-connectivity (up/down/left/right).  Returns grid unchanged if the
    target cell already has new_color.
    """
    result = grid.copy()
    target = int(result[row, col])
    if target == new_color:
        return result

    rows, cols = result.shape
    stack = [(row, col)]
    while stack:
        r, c = stack.pop()
        # Skip out-of-bounds or already-recolored cells
        if r < 0 or r >= rows or c < 0 or c >= cols:
            continue
        if result[r, c] != target:
            continue
        result[r, c] = new_color
        stack.extend([(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)])
    return result


# ---------------------------------------------------------------------------
# Object detection
# ---------------------------------------------------------------------------

def find_objects(grid: Grid, background: int | None = None) -> list[dict]:
    """Find connected foreground objects in the grid.

    Uses depth-first search with 4-connectivity to label connected components.
    Cells with the background colour are ignored.

    Args:
        grid:       The input grid.
        background: The colour to treat as background.  If None, the most
                    frequent colour is used (see background_color()).

    Returns:
        A list of object dicts, each containing:
            color    — colour value of the object
            pixels   — list of (row, col) tuples for every cell in the object
            bbox     — (r_min, c_min, r_max, c_max)  (inclusive)
            subgrid  — smallest Grid that fits the object (background cells = 0)
    """
    if background is None:
        background = background_color(grid)

    objects = []
    visited = np.zeros_like(grid, dtype=bool)

    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            color = int(grid[r, c])
            if color == background or visited[r, c]:
                continue

            # DFS to collect all cells connected to (r, c) with the same colour
            pixels: list[tuple[int, int]] = []
            stack = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
                    continue
                if visited[cr, cc] or grid[cr, cc] != color:
                    continue
                visited[cr, cc] = True
                pixels.append((cr, cc))
                stack.extend([(cr + 1, cc), (cr - 1, cc), (cr, cc + 1), (cr, cc - 1)])

            if not pixels:
                continue

            # Compute the tight bounding box from the collected pixel coords
            row_coords = [p[0] for p in pixels]
            col_coords = [p[1] for p in pixels]
            r_min, r_max = min(row_coords), max(row_coords)
            c_min, c_max = min(col_coords), max(col_coords)

            # Build a minimal subgrid containing just this object
            subgrid = np.zeros((r_max - r_min + 1, c_max - c_min + 1), dtype=np.int32)
            for pr, pc in pixels:
                subgrid[pr - r_min, pc - c_min] = color

            objects.append(
                {
                    "color": color,
                    "pixels": pixels,
                    "bbox": (r_min, c_min, r_max, c_max),
                    "subgrid": subgrid,
                }
            )

    return objects


def bounding_box(grid: Grid, color: int | None = None) -> tuple[int, int, int, int]:
    """Return (r_min, c_min, r_max, c_max) of non-background (or specific color) cells.

    If color is given, finds the bounding box of that colour only.
    Otherwise finds the bounding box of all non-background cells.
    Coordinates are inclusive on both ends.
    """
    if color is not None:
        mask_arr = grid == color
    else:
        bg = background_color(grid)
        mask_arr = grid != bg

    # Collapse each axis to find which rows/cols contain a matching cell
    rows = np.any(mask_arr, axis=1)
    cols = np.any(mask_arr, axis=0)

    # Guard: color absent from grid (or all-background grid with no color arg)
    if not np.any(rows):
        return 0, 0, 0, 0

    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    return int(r_min), int(c_min), int(r_max), int(c_max)


def crop_to_content(grid: Grid) -> Grid:
    """Crop the grid to the tight bounding box of all non-background content.

    Returns the original grid unchanged if the grid is entirely background.
    """
    bg = background_color(grid)
    if (grid == bg).all():
        return grid.copy()
    r_min, c_min, r_max, c_max = bounding_box(grid)
    return crop(grid, r_min, c_min, r_max + 1, c_max + 1)
