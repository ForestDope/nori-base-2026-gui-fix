# Triangle Soup

Differentiable inverse rendering with semi-transparent emissive triangles, built on [Mitsuba 3](https://mitsuba.readthedocs.io/) and [Dr.Jit](https://drjit.readthedocs.io/).

## Setup

```bash
pixi install
```

## Quick start

```bash
# Render orbit datasets for the provided scenes.
pixi run make-datasets      # cube + lego + fox + macaw
pixi run make-lego-dataset  # or just one

# Train end-to-end (deterministic view poses, tuned 3000-iter schedule).
pixi run train-lego
pixi run train-fox
pixi run train-macaw
```

Every `train-*` task writes a `progress_grid.png` into `output/<experiment>/` showing a 3×3 grid of renders from a fixed test view, sampled across the optimisation, with the ground truth in the last tile.

## CPU training (~5 minutes)

If CUDA is unavailable, use the `-cpu` variants. They converge on a modest orbit budget with Mitsuba's LLVM backend:

```bash
pixi run train-lego-cpu
pixi run train-fox-cpu
pixi run train-macaw-cpu
```

## Project structure

```
trianglesoup/
  rendering/
    integrator.py      # Emissive triangle soup integrator (forward + PRB backward)
    triangle_model.py  # Per-triangle attributes: color, occupancy, sigma, window function
    losses.py          # Photometric and regularization losses
  training/
    optimizer.py       # Training loop, pruning, densification
    variables.py       # Differentiable parameter management
    config.py          # CLI argument parsing and configuration
  train.py             # Entry point (also writes progress_grid.png)
data/
  generate_dataset.py  # Render deterministic orbit views from a Mitsuba scene
  generate_cube_dataset.py
  scenes/              # {cube,lego,fox,macaw}/scene.xml + meshes/
```
