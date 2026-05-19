# Round 188 Summary — H1.422: Per-Object CG on Long-Horizon Tasks

**Experiment**: H1.422 — Per-Object CG on Multi-Step Long-Horizon Manipulation (25 timesteps)

**Result**: **REFUTED**

**What we tested**: Whether Per-Object CG's +10.65% advantage over 2-Node CG (found at seq_len=10 in H1.421) scales to longer 25-timestep multi-step manipulation tasks (pick→move→place).

**Key numbers**:
- Baseline MLP: MSE 0.009335
- 2-Node CG: MSE 0.009088 (+2.64% vs baseline)
- Per-Object CG: MSE 0.009109 (+2.41% vs baseline, **-0.23% vs 2-Node CG**)

**What this means**: The per-object node structure's benefit is **sequence-length dependent**. At short horizons (10 timesteps), per-object tracking gives a clear +10.65% edge. At long horizons (25 timesteps), the advantage completely reverses — the simpler 2-Node abstraction is slightly more robust. This suggests that on longer sequences, fewer parameters to track across time reduces error accumulation, outweighing the benefits of fine-grained object representation.

**Next step (H1.423)**: Find the crossover point — test at intermediate sequence lengths (15, 20 timesteps) to map where the advantage flips, and explore whether adding temporal attention to Per-Object CG recovers the advantage on longer sequences.
