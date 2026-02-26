# ARC-AGI Solver

A multi-agent system for solving [ARC-AGI](https://arcprize.org/) (Abstraction and Reasoning Corpus) puzzles using large language models, with per-role model configuration so the best model type is used for each cognitive task.

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

---

## Architecture Overview

```
 ┌──────────────────────────────────────────────────────────┐
 │                     ARC-AGI Solver                       │
 │                                                          │
 │  ┌────────────────┐   think tags   ┌─────────────────┐  │
 │  │  Hypothesizer  │ ─────stripped──►     Coder        │  │
 │  │                │                │                  │  │
 │  │ deepseek-r1    │  clean rule    │ qwen2.5-coder   │  │
 │  │ (reasoning)    │ ─────────────► │ (code-focused)  │  │
 │  └────────────────┘                └────────┬────────┘  │
 │                                             │ code       │
 │  ┌────────────────┐   route+feedback  ┌────▼────────┐   │
 │  │     Critic     │ ◄─────────────────│  Sandbox    │   │
 │  │                │                   │  Evaluator  │   │
 │  │ deepseek-r1    │                   └─────────────┘   │
 │  │ (reasoning)    │                                      │
 │  └────────────────┘                                      │
 └──────────────────────────────────────────────────────────┘
```

The key insight: **match model strengths to cognitive tasks**.
- Reasoning models (`deepseek-r1`) excel at spatial pattern analysis and failure diagnosis
- Coding models (`qwen2.5-coder`) produce reliable, correctly formatted code without hedging

---

## Multi-Agent State Machine

### Diagram

```
  INPUT: ARC task (train pairs + test input)
         │
         ▼
  ╔══════════════════════════════╗
  ║      HYPOTHESIZING           ║
  ║  Hypothesizer studies grids  ║
  ║  → proposes up to 3 rules    ║
  ║  (reasoning model)           ║
  ╚══════════════╤═══════════════╝
                 │  strip <think> tags
                 │  parse & filter hypotheses
                 │
                 │  ┌─────────────────────────────────────────┐
                 │  │  for each hypothesis (outer loop)       │
                 ▼  ▼                                         │
  ╔══════════════════════════════╗                            │
  ║          CODING              ║◄── Critic feedback ──┐    │
  ║  Coder receives:             ║    (coder route)     │    │
  ║  · clean hypothesis          ║                      │    │
  ║  · training pairs (context)  ║                      │    │
  ║  · temperature (ramps up)    ║                      │    │
  ║  (coding model)              ║                      │    │
  ╚══════════════╤═══════════════╝                      │    │
                 │  code extracted via                  │    │
                 │  4-step cascade                      │    │
                 ▼                                      │    │
  ╔══════════════════════════════╗                      │    │
  ║         EVALUATING           ║                      │    │
  ║  Sandbox runs code against   ║                      │    │
  ║  every training pair         ║                      │    │
  ╚═══╤══════════════════════╤══╝                      │    │
      │ all correct          │ some wrong               │    │
      ▼                      ▼                          │    │
  ╔═══════════╗   ╔══════════════════════════════╗      │    │
  ║ CANDIDATE ║   ║       CRITICIZING            ║      │    │
  ║  saved    ║   ║  Critic diagnoses root cause ║      │    │
  ╚═════╤═════╝   ║  (reasoning model)           ║      │    │
        │         ╚══════════════╤═══════════════╝      │    │
        │              ┌─────────┴──────────┐           │    │
        │          flawed rule          code bug        │    │
        │              │                    └───────────┘    │
        │         abandon hyp                                │
        │              └────────────────────────────────────┘
        │
        ▼
  OUTPUT: candidate programs + best partial code
```

### Pseudocode

```
function solve(task):
    task_description   = format_grid_pairs(task.train)
    training_context   = format_training_examples(task.train)

    # ── Hypothesizing ────────────────────────────────────────────────────
    raw_response = Hypothesizer.generate(task_description)
    hypotheses   = parse_and_filter(raw_response, max_n=3)
    #   → paragraph-level split, 80-char noise filter, capped at 3

    candidates = []

    # ── Outer loop: one pass per hypothesis ──────────────────────────────
    for hypothesis in hypotheses:
        prev_n_correct   = -1
        no_improve_count = 0
        coder_feedback   = None

        # ── Inner loop: Coder attempts ───────────────────────────────────
        for attempt in 1 .. max_retries + 1:
            is_last    = (attempt == max_retries + 1)
            temperature = min((attempt - 1) × 0.3, 0.8)   # 0.0 → 0.3 → 0.6

            clean_hyp  = strip_think_tags(hypothesis)      # remove <think>…</think>
            code_text  = Coder.generate(clean_hyp, coder_feedback,
                                        training_context, temperature)
            code       = extract_code(code_text)           # 4-step cascade

            if code is None:
                break    # no_code_block → abandon hypothesis

            result = Sandbox.evaluate(code, task.train)

            if result.all_correct:
                candidates.append(code)
                break    # hypothesis solved

            # Stuck detection: 0 correct for 2+ consecutive attempts
            if result.n_correct <= prev_n_correct:
                no_improve_count += 1
            else:
                no_improve_count = 0
            prev_n_correct = result.n_correct

            stuck = (result.n_correct == 0 and no_improve_count >= 2)
            if is_last or stuck:
                break    # Critic can't help here

            diagnosis = Critic.analyze(hypothesis, code, result)
            if diagnosis.route == HYPOTHESIZER:
                break    # logic flaw → try next hypothesis
            else:
                coder_feedback = diagnosis.feedback   # code bug → retry

    return first_candidate or best_partial_code
```

### Code Extraction Cascade

When the model response arrives, code is found via a four-step fallback:

```
  model response text
         │
         ├─1─ ```python … ``` found?   ─yes─► extract + truncate prose
         │       (flexible whitespace)
         ├─2─ ``` … ``` with def?      ─yes─► extract + truncate prose
         │
         ├─3─ bare "def transform(" ?  ─yes─► extract from def onwards
         │
         ├─4─ "import numpy" + "def "? ─yes─► extract from import onwards
         │       (reasoning-only mode)
         │
         └──── None → no_code_block → abandon hypothesis
```

---

## Single-Agent Baseline

A simpler self-correction loop used as a comparison baseline.

```
  task (train pairs + test input)
         │
         ▼
  LLM writes def transform(grid) → grid
         │
         ▼
  Execute against all training pairs
         │
    ┌────┴──────────────────────┐
    │ all correct?               │ no
    └──────────────────────────┬┘
         │ yes                  │
         ▼                      ▼
  apply to test input    format pixel-level diff
         │                      │
         ▼                      └──► retry (↑ temperature: 0.0→0.4→0.8→1.0)
  predicted output
```

---

## Model Selection Strategy

Each agent role has a different cognitive requirement, so a different model type is optimal:

| Role | Best Model Type | Example | Why |
|------|----------------|---------|-----|
| **Hypothesizer** | Reasoning / thinking | `deepseek-r1:32b` | Needs spatial pattern analysis and lateral thinking across grid examples |
| **Coder** | Code-focused | `qwen2.5-coder:7b` | Needs reliable `\`\`\`python` output, not reasoning; small + fast |
| **Critic** | Reasoning | `deepseek-r1:8b` | Needs to diagnose logic vs. implementation bugs from diffs |

The Coder prompt is intentionally **non-reasoning**: it opens with "Implement it immediately. Start with ` ```python `." No "think step by step." The reasoning has already been done by the Hypothesizer; the Coder's job is pure translation.

The Hypothesizer's `<think>` chain-of-thought is **stripped** before being handed to the Coder — the small coding model only sees the clean, structured algorithm description.

---

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
│   ├── roles.py                # Hypothesizer, Coder, Critic agent roles + prompts
│   ├── orchestrator.py         # State-machine orchestrator (primary solver)
│   ├── multi_agent.py          # Shared helpers + cycle-based MultiAgent class
│   ├── ensemble.py             # Run multiple Orchestrators, majority vote
│   └── single_agent.py         # Single-agent baseline
├── tests/                      # pytest suite (412 tests, ~92% coverage)
├── data/                       # ARC dataset (400 training + 400 evaluation tasks)
├── run_multi_agent.py          # CLI for the multi-agent Orchestrator
├── run_baseline.py             # CLI for the single-agent baseline
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

For the Anthropic backend:

```bash
export ANTHROPIC_API_KEY=sk-...
```

For the Ollama backend, install [Ollama](https://ollama.com/) and pull models:

```bash
ollama pull deepseek-r1:32b      # Hypothesizer + Critic (reasoning)
ollama pull deepseek-r1:8b       # Critic (lighter option)
ollama pull qwen2.5-coder:7b     # Coder (fast, code-focused)
```

---

## Usage

### Multi-Agent Orchestrator (CLI)

```bash
# Recommended: per-role models
python run_multi_agent.py \
    --task data/data/training/007bbfb7.json \
    --hypothesizer-model deepseek-r1:32b \
    --coder-model qwen2.5-coder:7b \
    --critic-model deepseek-r1:8b

# Run over a directory (reports train-solved + test-correct rates)
python run_multi_agent.py \
    --dir data/data/training \
    --limit 20 \
    --hypothesizer-model deepseek-r1:32b \
    --coder-model qwen2.5-coder:7b \
    --critic-model deepseek-r1:8b

# Single model for all roles (quick experiments)
python run_multi_agent.py \
    --task data/data/training/007bbfb7.json \
    --model deepseek-r1:8b

# Anthropic backend
python run_multi_agent.py \
    --task data/data/training/007bbfb7.json \
    --backend anthropic
```

#### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--task FILE` | — | Single ARC task JSON file |
| `--dir DIR` | — | Directory of task JSON files |
| `--limit N` | none | Max tasks to run (directory mode) |
| `--backend` | `ollama` | `ollama` or `anthropic` |
| `--model` | backend default | Fallback model for all roles |
| `--hypothesizer-model` | `--model` | Model for the Hypothesizer |
| `--coder-model` | `--model` | Model for the Coder |
| `--critic-model` | `--model` | Model for the Critic |
| `--n-hypotheses N` | `3` | Hypotheses to generate per task |
| `--max-retries N` | `2` | Extra Coder attempts per hypothesis |
| `--timeout SECS` | `300` | Seconds per LLM call (Ollama) |
| `--debug` | off | Print per-state diagnostic output |
| `--quiet` | off | Suppress per-task output |

### Multi-Agent Orchestrator (Python API)

```python
import json
import numpy as np
from agents.orchestrator import Orchestrator

with open("data/data/training/007bbfb7.json") as f:
    task = json.load(f)

for split in ("train", "test"):
    for pair in task.get(split, []):
        for key in ("input", "output"):
            if key in pair:
                pair[key] = np.array(pair[key], dtype=np.int32)

orch = Orchestrator(
    backend="ollama",
    hypothesizer_model="deepseek-r1:32b",   # reasoning model
    coder_model="qwen2.5-coder:7b",          # coding model
    critic_model="deepseek-r1:8b",           # reasoning model
    n_hypotheses=3,
    max_retries=2,
    timeout=300,
    debug=True,
)

result = orch.solve(task)
print(result["success"])           # True if any candidate passed all training pairs
print(result["candidates"])        # programs that passed (code, hypothesis, n_correct)
print(result["code"])              # best code found (candidate or best partial)
print(result["test_correct"])      # True/False/None — held-out test accuracy
```

### Single-Agent Baseline (CLI)

```bash
python run_baseline.py --task data/data/training/007bbfb7.json

python run_baseline.py --dir data/data/training --limit 20

python run_baseline.py --task data/data/training/007bbfb7.json \
    --backend anthropic --retries 5 --debug
```

| Flag | Default | Description |
|------|---------|-------------|
| `--task` | — | Single task JSON file |
| `--dir` | — | Directory of task JSON files |
| `--limit` | none | Max tasks (directory mode) |
| `--backend` | `ollama` | `ollama` or `anthropic` |
| `--model` | backend default | Model name override |
| `--retries` | `3` | Self-correction attempts |
| `--timeout` | `600` | Seconds per LLM call (Ollama) |
| `--quiet` | off | Suppress per-task output |
| `--debug` | off | Print raw model responses |

Results are saved as JSON under `logs/`.

---

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

Generated code runs in an isolated child process (`multiprocessing.Process`) that is force-killed after 10 seconds. The child's namespace contains only the DSL primitives and numpy — no filesystem, network, or `input()` access. Code referencing `sys.stdin` or `input` is rejected before the subprocess is even spawned.

### LLM Client (`agents/llm_client.py`)

A thin backend-agnostic wrapper. All agents call `generate(system_prompt, messages, temperature) → str`.

**Ollama** — streams NDJSON from the local `/api/chat` endpoint via `urllib`.
- Thinking tokens (`deepseek-r1`) arrive in a separate `thinking` field.
- When content is non-empty: thinking is wrapped in `<think>…</think>` and prepended.
- When content is empty (model put everything in thinking): raw thinking is returned without tags so code extractors can find `def transform` or ` ```python ` blocks within it.

**Anthropic** — uses the `anthropic` SDK, single non-streaming call.

Default models: `deepseek-r1:32b` (Ollama), `claude-sonnet-4-6` (Anthropic).

### Per-Role Model Configuration

Each of `Orchestrator`, `MultiAgent`, and `Ensemble` accepts independent model parameters:

```
                 ┌─ hypothesizer_model ─► Hypothesizer LLMClient
                 │
Orchestrator ────┼─ coder_model ────────► Coder LLMClient
                 │
                 └─ critic_model ────────► Critic LLMClient

  model= (optional fallback applied to any role not given its own model)
```

All three share the same `backend` and `timeout`. Mixing backends per role is not currently supported.

### Agent Roles & Prompt Design (`agents/roles.py`)

#### Hypothesizer
Proposes up to 3 distinct natural-language transformation algorithms from the training pairs. Prompt instructs: no code, numbered hypotheses, precise about colours and geometry.

#### Coder
Translates one clean hypothesis into a `def transform(grid)` function using DSL primitives.

Prompt is **intentionally terse and imperative** — designed for non-reasoning coding models:
- Opens with: *"Implement it immediately. Start your response with \`\`\`python."*
- Closes with: *"No explanation. No commentary. No prose."*
- Contains an explicit sequential-overwrite warning with bad/good examples:

```
BAD  — mutates grid in place (second assignment sees first's changes):
    grid[grid == 1] = 2
    grid[grid == 2] = 1   ← also changes the 1s just set to 2

GOOD — freeze source with np.copy:
    new = np.copy(grid)
    new[grid == 1] = 2
    new[grid == 2] = 1

GOOD — single vectorised step with np.select:
    new = np.select([grid==1, grid==2], [2, 1], default=grid)
```

Before building the user message, `Coder.generate()` strips `<think>…</think>` blocks from the hypothesis. This ensures the small coding model receives only the clean algorithm description, not the reasoning model's chain-of-thought.

#### Critic
Analyzes a failed attempt (hypothesis + code + error + pixel diff showing full grids for the first failing pair) and outputs a route + actionable feedback:
- `ROUTE: HYPOTHESIZER` — the logical rule itself is wrong
- `ROUTE: CODER` — the hypothesis is sound but was implemented incorrectly

### Hypothesis Parsing

The Hypothesizer's response is parsed with a three-layer strategy tuned for reasoning models that embed numbered reasoning steps inside prose:

```
  raw Hypothesizer response
          │
          ├─1─ paragraph-level split (blank-line delimited)?
          │      → only captures numbered paragraphs, not inline sub-steps
          │
          ├─2─ line-level split fallback
          │      → for models that don't use blank lines between hypotheses
          │
          └─3─ noise filter: drop items < 80 chars (reasoning fragments)
                  + cap at n_hypotheses (default 3)
```

### Orchestrator Loop Details (`agents/orchestrator.py`)

**Temperature ramp** — greedy on first attempt, increasing on retries:

| Attempt | Temperature |
|---------|------------|
| 1 | 0.0 (greedy) |
| 2 | 0.3 |
| 3 | 0.6 |
| 4+ | 0.8 (cap) |

**Stuck detection** — if a hypothesis scores 0/N correct for 2+ consecutive attempts, the Critic is skipped and the hypothesis is abandoned. A hypothesis that never gets any pair right has a logic flaw, not a code bug.

**Last-attempt guard** — the Critic is never called on the final attempt (its feedback would be discarded). This saves one LLM call per hypothesis.

**Training context** — the Coder receives the full training pairs alongside the hypothesis so it can mentally verify its implementation against concrete examples.

### Ensemble (`agents/ensemble.py`)

Runs multiple independent Orchestrator instances on the same task, collects every program that passes all training pairs, then selects the test-input output by majority vote. Statistically, if k independent code paths agree on the same pixel-perfect output, it is very likely to be the correct generalised rule.

---

## Running Tests

```bash
pytest tests/
```

412 tests covering:
- DSL primitives and edge cases (all-background grids, large inputs)
- Sandbox execution (timeouts, exceptions, stdin rejection)
- LLM client streaming (thinking tokens, partial results, timeouts)
- All three agent roles (system prompt content, think-stripping, routing)
- Orchestrator state transitions (stuck detection, last-attempt guard, retry loop)
- Hypothesis parsing (paragraph split, noise filter, max_n cap)
- Code extraction (all four fallback levels)
- Ensemble voting and majority selection
- Single-agent baseline (code extraction, retry loop, logging)

All network calls are fully mocked — no Ollama or Anthropic connection needed to run the suite.
