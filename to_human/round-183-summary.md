# Round 183 Summary — Temporal CG: Extended Training, Proper GRU, Curriculum Learning

## What We Did

Following H1.414's finding that Temporal CG catastrophically failed (losses 10-100x worse than baseline), we tested three fixes: (1) **H1.415**: extended training from 40 to 100 epochs, (2) **H1.416**: replacing the self-recurrent GRU with a proper GRU using a state_dynamics network for meaningful input, and (3) **H1.417**: curriculum learning that gradually increases sequence length (3→5→10 steps).

## Results

Extended training (H1.415) dramatically improved Temp-CG from catastrophic failure to competitive — final val loss dropped from 0.258 (H1.414, 1-step) to 0.033 (H1.415, 100ep). However, **no Temp-CG variant beat the simple MLP baseline** (0.0246). The proper GRU (H1.416) actually performed worse than the self-recurrent version (0.0365 vs 0.0334), and curriculum learning (H1.417) was worst overall (0.0393) though it achieved the best 5-step performance (0.0307) among Temp-CG variants.

## Key Insight

The Temporal CG architecture is **fundamentally mismatched** for this prediction task. Neither more training, better GRU formulation, nor curriculum scheduling recovers baseline performance. The inverse loss scaling pattern persists across all variants (lowest loss at 5 steps, highest at 1 and 10 steps), suggesting the recurrent architecture is optimized for medium-length sequences but fails at extremes. The original CG also underperformed baseline in this scaled-down configuration (0.0402 vs 0.0246), possibly due to reduced unified space or the task being too simple for CG's inductive biases.

## Next Steps

H1.418: Test Temp-CG on tasks with stronger temporal dependencies (physics with momentum/inertia) where recurrence should theoretically help. Alternatively, explore transformer-based temporal modeling as a replacement for recurrent message passing.
