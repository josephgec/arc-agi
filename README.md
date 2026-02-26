# ARC-AGI Solver

A multi-agent system for solving [ARC-AGI](https://arcprize.org/) (Abstraction and Reasoning Corpus) puzzles using large language models.

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

Two solvers are implemented, sharing the same DSL and execution sandbox.

### Multi-Agent System (primary)

Three specialised agents collaborate in a state-machine loop:

- **Hypothesizer** — studies the training grid pairs and proposes up to 3 competing natural-language transformation algorithms. Does not write code.
- **Coder** — translates one hypothesis into an executable `transform(grid)` function using only the DSL primitives.
- **Critic** — diagnoses failures and routes blame: either the logical rule was wrong (→ try next hypothesis) or the implementation had a bug (→ retry the same hypothesis with targeted feedback).

```
                    ┌─────────────────────────────┐
                    │        HYPOTHESIZING         │
                    │  Hypothesizer proposes 1–3   │
                    │  natural-language algorithms  │
                    └──────────────┬───────────────┘
                                   │  for each hypothesis
                    ┌──────────────▼───────────────┐
                    │           CODING             │
                    │  Coder writes transform(grid) │◄─── Critic feedback (coder route)
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │          EVALUATING          │
                    │  Run against all train pairs  │
                    └──────┬───────────────────────┘
                           │                       │
                     all correct               some wrong
                           │                       │
               ┌───────────▼──────┐   ┌────────────▼────────────┐
               │    CANDIDATE     │   │       CRITICIZING        │
               │  solution saved  │   │  Critic diagnoses cause  │
               └──────────────────┘   └─────┬──────────┬─────────┘
                                            │          │
                                     flawed rule    code bug
                                            │          │
                                      next hyp     retry Coder
                                                  (↑ temperature)
```

**Stuck detection** — if a hypothesis produces 0 correct pairs across 2+ consecutive attempts, it is abandoned early without calling the Critic (the hypothesis is clearly wrong, not just poorly coded).

**Training context** — the Coder receives the full set of training grid pairs alongside the hypothesis so it can verify its implementation mentally before returning code.

**Temperature diversity** — the Coder's temperature ramps up with each retry (0.0 → 0.3 → 0.6) to encourage diverse implementations rather than identical repetitions of the same bug.

### Single-Agent Baseline

A simpler self-correction loop where one LLM writes code and re-tries with pixel-level error feedback. Used as a baseline and for quick experimentation.

```
Task prompt (training pairs)
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
├── arc/                        # Core library (backend-agnostic)
│   ├── grid.py                 # Grid type, task loading, utility functions
│   ├── dsl.py                  # Transformation primitives (DSL)
│   ├── sandbox.py              # Isolated subprocess execution of generated code
│   ├── evaluate.py             # Evaluation harness and scoring
│   └── visualize.py            # Terminal (ANSI) and matplotlib visualisation
├── agents/
│   ├── llm_client.py           # Unified LLM backend (Ollama + Anthropic)
│   ├── roles.py                # Hypothesizer, Coder, Critic agent roles
│   ├── orchestrator.py         # State-machine orchestrator (primary solver)
│   ├── multi_agent.py          # Cycle-based multi-agent loop (alternative)
│   ├── ensemble.py             # Run multiple Orchestrators and vote
│   └── single_agent.py         # Single-agent baseline
├── tests/                      # pytest test suite (400 tests, ~92% coverage)
├── data/                       # ARC dataset (400 training + 400 evaluation tasks)
├── run_baseline.py             # CLI entry point for the single-agent baseline
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

For the Anthropic backend, set your API key:

```bash
export ANTHROPIC_API_KEY=sk-...
```

For the Ollama backend, install [Ollama](https://ollama.com/) and pull a model:

```bash
ollama pull deepseek-r1:32b      # strong reasoning, best for Hypothesizer
ollama pull qwen2.5-coder:7b     # fast, reliable code output, good for Coder
```

## Usage

### Multi-Agent Orchestrator

```python
import json
import numpy as np
from agents.orchestrator import Orchestrator

with open("data/data/training/007bbfb7.json") as f:
    task = json.load(f)

# Convert lists to numpy arrays
for split in ("train", "test"):
    for pair in task.get(split, []):
        for key in ("input", "output"):
            if key in pair:
                pair[key] = np.array(pair[key], dtype=np.int32)

orch = Orchestrator(
    backend="ollama",
    model="deepseek-r1:32b",
    n_hypotheses=3,   # hypotheses to generate per cycle
    max_retries=2,    # additional Coder attempts per hypothesis
    timeout=300,      # seconds per LLM call
    debug=True,
)

result = orch.solve(task)
print(result["success"])      # True if any hypothesis passed all training pairs
print(result["candidates"])   # all programs that passed training
print(result["code"])         # best code found
```

### Single-Agent Baseline (CLI)

```bash
# Solve a single task
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

#### CLI Options

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

Generated functions run inside a namespace pre-populated with transformation primitives — no imports needed. All primitives return new arrays (immutable).

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

`numpy` is also available as `np` inside every generated function.

### Execution Sandbox (`arc/sandbox.py`)

Generated code runs in an isolated child process (`multiprocessing.Process`) that is force-killed after 10 seconds. The child's namespace contains only the DSL primitives and numpy — no file system, network, or `input()` access. Code that references `sys.stdin` or `input` is rejected before the subprocess is even spawned.

### LLM Client (`agents/llm_client.py`)

A thin backend-agnostic wrapper that all agents call via `generate(system_prompt, messages, temperature)`.

- **Ollama** — streams NDJSON from the local `/api/chat` endpoint via `urllib`. Thinking tokens from extended-reasoning models (e.g. `deepseek-r1`) arrive in a separate `thinking` field; when content is non-empty, they are wrapped in `<think>…</think>` and prepended so callers can strip them uniformly. When content is empty (the model placed its entire response in the thinking field), the raw thinking is returned without tags so code extractors can still find `def transform` or ` ```python ` blocks within it.
- **Anthropic** — uses the `anthropic` SDK with a single non-streaming call.

Default models: `deepseek-r1:32b` (Ollama), `claude-sonnet-4-6` (Anthropic).

### Agent Roles (`agents/roles.py`)

| Role | Responsibility |
|------|---------------|
| `Hypothesizer` | Proposes 3 distinct natural-language transformation algorithms from the training pairs. Never writes code. |
| `Coder` | Translates one hypothesis into a `def transform(grid)` function using DSL primitives. Accepts Critic feedback and training examples for context. |
| `Critic` | Analyzes a failed attempt (hypothesis + code + error + pixel diff) and routes blame: `ROUTE: HYPOTHESIZER` (wrong logic) or `ROUTE: CODER` (implementation bug). |

### Hypothesis Parsing

The Hypothesizer's response is parsed with a three-layer strategy tuned for reasoning models that embed numbered lists inside their chain-of-thought:

1. **Paragraph-level split** — preferred; only captures items that begin a blank-line-separated paragraph with a digit+period, avoiding sub-steps inside prose.
2. **Line-level split** — fallback for models that don't add blank lines between hypotheses.
3. **Noise filter** — items shorter than 80 characters are discarded as reasoning fragments, not full algorithm descriptions.

Results are capped at `n_hypotheses` (default 3).

### Orchestrator (`agents/orchestrator.py`)

The `Orchestrator` is the primary solver. It owns task state and drives a two-level loop:

- **Outer loop** — one pass per hypothesis produced by the Hypothesizer.
- **Inner loop** — up to `max_retries + 1` Coder attempts per hypothesis.

A `CANDIDATE` is recorded whenever code passes all training pairs; the outer loop continues to find additional candidates from remaining hypotheses.

### Ensemble (`agents/ensemble.py`)

`Ensemble` runs multiple independent `Orchestrator` instances on the same task and selects the output by majority vote. Useful when a single run is unreliable.

## Running Tests

```bash
pytest tests/
```

400 tests covering DSL primitives, sandbox execution, LLM client streaming, all three agent roles, orchestrator state transitions, ensemble voting, and the single-agent baseline. Network calls are fully mocked.
