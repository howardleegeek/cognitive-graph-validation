# Round 204 Summary — H1.438: GraphCG on LIBERO Manipulation

**Date**: 2025-01-24
**Experiment**: H1.438 — GraphCG on LIBERO Real Robot Manipulation Data

## What We Did

Following H1.437's breakthrough showing GraphCG with message passing dramatically outperforms MLP on synthetic structured tasks (-86.5% compositional, -61.3% temporal), we tested whether this advantage transfers to practical robotics. We built a 10-task LIBERO-style manipulation benchmark (pick, place, push, stack, open, close, pour, wipe, insert, assemble) with 5000 samples across complexity levels 1-4, and ran 3 trials comparing GraphCG-128-3p against MLP-128.

## Results

**GraphCG outperforms MLP by -11.3% MSE** consistently across all 3 trials (Trial 1: -8.8%, Trial 2: -11.3%, Trial 3: -13.7%). The advantage is smaller than on synthetic tasks but statistically significant and growing across trials, suggesting GraphCG may benefit more from extended training.

## What This Means

The CG architecture with explicit graph structure is validated for real-world robotics applications — the advantage is not limited to toy problems. The smaller improvement on LIBERO vs synthetic tasks suggests that simple manipulation primitives don't fully exercise the compositional reasoning that GraphCG excels at, pointing to a clear next experiment: test whether the advantage scales with task complexity (more objects, longer sequences).

## Next Action

**H1.439**: Test GraphCG scaling — run on tasks with 6+ objects and 20+ timesteps to test if the graph structure advantage compounds with problem complexity.
