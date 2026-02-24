# ARC-AGI Solver

A single-agent baseline for solving [ARC-AGI](https://arcprize.org/) (Abstraction and Reasoning Corpus) puzzles using large language models.

## What is ARC-AGI?

ARC-AGI tasks are pattern-recognition puzzles. Each task presents a few input→output grid pairs that share a hidden transformation rule. The goal is to infer the rule and apply it to a new test input.

Grids are 2-D arrays of integers 0–9, where each integer represents a colour:

| Index | Colour  |
|-------|---------|
| 0     | black   |
| 1     | blue    |
| 2     | red     |
| 3     | green   |
| 4     | yellow  |
| 5     | grey    |
| 6     | magenta |
| 7     | orange  |
| 8     | azure   |
| 9     | maroon  |

## Approach

The solver prompts an LLM to write a Python `transform` function, executes it against all training pairs, and iteratively feeds back pixel-level error details if any pair is wrong. This self-correction loop runs up to `max_retries` times with increasing temperature to encourage diverse solutions.

```
Task prompt (training pairs + test input)
        │
        ▼
   LLM generates def transform(grid) → grid
        │
        ▼
   Execute against all training pairs
        │
   ┌────┴─────────────────┐
   │ all correct?          │ no → format pixel-level errors → retry (↑ temperature)
   └──────────────────────┘
        │ yes
        ▼
   Apply to test input → predicted output
```

## Project Structure

```
arc-agi/
├── arc/                     # Core library
│   ├── grid.py              # Grid type, task loading, utility functions
│   ├── dsl.py               # Transformation primitives (DSL)
│   ├── evaluate.py          # Evaluation harness and scoring
│   └── visualize.py         # Terminal (ANSI) and matplotlib visualisation
├── agents/
│   └── single_agent.py      # LLM-based solving agent
├── tests/                   # pytest test suite
├── run_baseline.py          # CLI entry point
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

For the Anthropic backend, set your API key:

```bash
echo "ANTHROPIC_API_KEY=sk-..." > .env
```

For the Ollama backend, install [Ollama](https://ollama.com/) and pull a model:

```bash
ollama pull deepseek-r1:8b
```

## Usage

```bash
# Solve a single task (Ollama/deepseek-r1 by default)
python run_baseline.py --task data/data/training/007bbfb7.json

# Solve the first 20 tasks in a directory
python run_baseline.py --dir data/data/training --limit 20

# Use Anthropic Claude instead
python run_baseline.py --task data/data/training/007bbfb7.json --backend anthropic

# Use a different local model
python run_baseline.py --task data/data/training/007bbfb7.json --model qwen2.5-coder:32b

# More retries, debug output
python run_baseline.py --task data/data/training/007bbfb7.json --retries 5 --debug
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--task` | — | Path to a single task JSON file |
| `--dir` | — | Directory of task JSON files |
| `--limit` | none | Maximum number of tasks to run |
| `--backend` | `ollama` | `ollama` or `anthropic` |
| `--model` | backend default | Model name override |
| `--retries` | `3` | Self-correction attempts after the first try |
| `--timeout` | `600` | Seconds per model call (Ollama only) |
| `--quiet` | off | Suppress per-task output |
| `--debug` | off | Print raw model responses |

Results are saved as JSON under `logs/`.

## Implementation Details

### DSL (`arc/dsl.py`)

To give the model a useful vocabulary, every generated function runs inside a namespace pre-populated with transformation primitives — no imports needed. All primitives are pure (immutable: they return a new array rather than mutating the input).

| Primitive | Description |
|-----------|-------------|
| `crop(grid, r1, c1, r2, c2)` | Extract a rectangular sub-grid |
| `rotate(grid, n)` | Rotate 90° counter-clockwise n times |
| `flip(grid, axis)` | Flip vertically (0) or horizontally (1) |
| `translate(grid, dr, dc, fill)` | Shift, filling vacated cells with `fill` |
| `scale(grid, factor)` | Expand each cell into a factor×factor block |
| `tile(grid, n_rows, n_cols)` | Repeat the grid in a grid pattern |
| `recolor(grid, from_c, to_c)` | Replace one colour with another |
| `mask(grid, mask_grid, fill)` | Zero out cells where mask == 0 |
| `overlay(base, top, transparent)` | Paint non-transparent cells of `top` onto `base` |
| `flood_fill(grid, row, col, color)` | 4-connected flood fill |
| `find_objects(grid, background)` | DFS connected-component detection |
| `bounding_box(grid, color)` | Tight bounding box of a colour or all foreground |
| `crop_to_content(grid)` | Crop to the bounding box of non-background content |

### Agent (`agents/single_agent.py`)

**Prompt construction** — Training pairs are serialised as Python list literals. For tasks with very large grids (> 300 total cells per pair), the number of pairs shown is capped at 2 to stay within context limits. When the output is an exact N×M tiling of input-sized blocks, a pre-computed block analysis is appended to help the model spot the pattern.

**Code extraction** — Model responses are searched for fenced `` ```python `` blocks. The *last* syntactically-valid block containing a `def` is used (reasoning models often draft multiple versions and refine toward the end). If trailing prose corrupts the block, it is trimmed line-by-line via AST validation. Blocks containing no function definition (e.g. grid literals) are rejected. A bare `def` outside any fence is used as a last resort.

**Execution sandbox** — Generated code runs under `exec()` in an isolated namespace seeded with the DSL and numpy. `stdout` is suppressed. Code that references `input()` or `sys.stdin` is rejected before execution. If the model names its function something other than `transform`, the last user-defined callable in the namespace is used as a fallback.

**Self-correction loop** — On each failed attempt, the agent sends back a message listing which pairs were wrong, the full expected and predicted grids, and a cell-level diff (coordinates + colour names). The temperature schedule is `[0.0, 0.4, 0.8, 1.0]` — greedy on the first try, increasingly random on retries.

**Backends:**

- **Ollama** — Calls the local Ollama `/api/chat` endpoint via `urllib` with streaming NDJSON. Exits the stream early once a complete code block is detected. Recovers partial results if the wall-clock deadline fires mid-stream. Thinking tokens from extended-reasoning models (e.g. deepseek-r1) are collected separately, wrapped in `<think>` tags, and stripped before code extraction.
- **Anthropic** — Uses the `anthropic` SDK with a single non-streaming call (`max_tokens=8192`).

### Evaluation (`arc/evaluate.py`)

`evaluate_task(task, transform_fn)` applies a function to every training pair, catching exceptions per-pair, and returns correctness stats. `score_all(tasks_dir, transform_fn)` runs this across a directory and reports task-level accuracy (all pairs correct) and pair-level accuracy separately.

## Running Tests

```bash
pytest tests/
```

The test suite covers:
- Grid utilities and DSL primitives (unit tests with numpy assertions)
- Evaluation harness (correct/wrong/exception cases, scoring)
- Visualisation (terminal output captured with `capsys`, matplotlib mocked)
- Agent internals (code extraction edge cases, Ollama streaming mocked via `urllib`, end-to-end solve loop with mocked Anthropic API)
