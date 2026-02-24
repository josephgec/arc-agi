"""Grid visualization: ANSI terminal output and matplotlib figures."""
from __future__ import annotations

import numpy as np
from .grid import Grid, COLOR_NAMES

# ---------------------------------------------------------------------------
# ANSI terminal colours
# Maps ARC colour index → ANSI escape code that sets that background colour.
# Uses 256-colour approximations for indices that have no standard 8-colour match.
# ---------------------------------------------------------------------------

_ANSI_BG = {
    0: "\033[40m",        # black
    1: "\033[44m",        # blue
    2: "\033[41m",        # red
    3: "\033[42m",        # green
    4: "\033[43m",        # yellow
    5: "\033[100m",       # grey  (bright-black background)
    6: "\033[45m",        # magenta
    7: "\033[48;5;208m",  # orange (256-colour)
    8: "\033[46m",        # cyan / azure
    9: "\033[48;5;88m",   # maroon (256-colour)
}
_RESET = "\033[0m"  # reset all terminal attributes


def print_grid(grid: Grid, label: str = "") -> None:
    """Print a grid to the terminal using ANSI background colours.

    Each cell is rendered as two spaces with the appropriate background colour.
    An optional label and grid dimensions are printed above the grid.
    """
    if label:
        print(f"  {label}  ({grid.shape[0]}×{grid.shape[1]})")
    for row in grid:
        line = ""
        for cell in row:
            bg = _ANSI_BG.get(int(cell), "\033[47m")  # fallback: white bg
            line += f"{bg}  {_RESET}"
        print(line)
    print()


def print_task(task: dict, max_pairs: int = 5) -> None:
    """Print all training pairs and test inputs to the terminal.

    Args:
        task:      ARC task dict (same format as load_task returns).
        max_pairs: Maximum number of training pairs to display.
    """
    for i, pair in enumerate(task["train"][:max_pairs]):
        print(f"=== Train pair {i} ===")
        print_grid(pair["input"], "input")
        print_grid(pair["output"], "output")
    for i, pair in enumerate(task["test"]):
        print(f"=== Test {i} ===")
        print_grid(pair["input"], "input")
        if "output" in pair:
            print_grid(pair["output"], "expected output")


# ---------------------------------------------------------------------------
# Matplotlib visualization
# ---------------------------------------------------------------------------

# Hex colours for matplotlib plots — one per ARC colour index 0–9.
_MPL_COLORS = [
    "#000000",  # 0 black
    "#0074D9",  # 1 blue
    "#FF4136",  # 2 red
    "#2ECC40",  # 3 green
    "#FFDC00",  # 4 yellow
    "#AAAAAA",  # 5 grey
    "#F012BE",  # 6 magenta
    "#FF851B",  # 7 orange
    "#7FDBFF",  # 8 azure
    "#870C25",  # 9 maroon
]


def plot_grid(grid: Grid, ax=None, title: str = "") -> None:
    """Plot a single grid on a matplotlib axis.

    Uses a fixed discrete colormap matching the ARC colour palette.
    Draws thin white grid lines between cells.  If ax is None, a new figure
    and axis are created automatically.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    cmap = mcolors.ListedColormap(_MPL_COLORS)
    bounds = list(range(11))  # boundary values: 0..10 → 10 colour buckets
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(grid, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)

    # Overlay thin white grid lines so individual cells are visible
    rows, cols = grid.shape
    for x in range(cols + 1):
        ax.axvline(x - 0.5, color="white", linewidth=0.5)
    for y in range(rows + 1):
        ax.axhline(y - 0.5, color="white", linewidth=0.5)


def plot_task(task: dict, title: str = "", save_path: str | None = None) -> None:
    """Plot all training pairs and test inputs side by side in a single figure.

    Layout: [train_0_in | train_0_out | train_1_in | train_1_out | … | test_0_in | …]

    Args:
        task:      ARC task dict (same format as load_task returns).
        title:     Optional figure-level title.
        save_path: If given, save the figure to this path instead of displaying it.
    """
    import matplotlib.pyplot as plt

    train = task["train"]
    test = task["test"]
    n_train = len(train)
    n_test = len(test)

    # Each training pair occupies 2 columns (input + output); test inputs get 1 each.
    n_cols = n_train * 2 + n_test
    fig, axes = plt.subplots(1, n_cols, figsize=(2 * n_cols, 3))
    if n_cols == 1:
        axes = [axes]

    col = 0
    for i, pair in enumerate(train):
        plot_grid(pair["input"],  axes[col], f"train {i} in");  col += 1
        plot_grid(pair["output"], axes[col], f"train {i} out"); col += 1
    for i, pair in enumerate(test):
        plot_grid(pair["input"],  axes[col], f"test {i} in");   col += 1

    if title:
        fig.suptitle(title, fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()
