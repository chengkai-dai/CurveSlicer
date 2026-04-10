# CurveSlicer

Curved slicing optimizer for DLP printing.

## Setup

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/). 

```bash
git clone https://github.com/chengkai-dai/CurveSlicer
cd CurveSlicer
uv sync
```

## Run

Default example:

```bash
uv run python run_optimization.py
```

Other presets:

```bash
uv run python run_optimization.py --model hook
uv run python run_optimization.py --model fertility
uv run python run_optimization.py --model woman-pully
```

## Useful args

- `--model`: `bunny`, `hook`, `woman-pully`, `fertility`
- `--mesh`: override the mesh path
- `--n-layers`: number of slicing layers
- `--lr`: learning rate
- `--iters`: max iterations per restart
- `--restarts`: number of restarts
- `--k` / `--k-start`: sharpness and optional annealing start
- `--setup-opt`: also optimize quaternion and translation
- `--output`: output directory
- `--log-file`: pass `''` to disable file logging
- `--quiet`: disable console logging

GUI is under development.
