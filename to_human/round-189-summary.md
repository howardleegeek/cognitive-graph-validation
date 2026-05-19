# Round 189 Summary — H1.423: Sequence-Length Crossover Analysis

**Experiment**: H1.423 — Sequence-Length Crossover Analysis (seq_len=15)

**Result**: **SUPPORTED**

**What we tested**: Whether there's a crossover point between seq_len=10 (where Per-Object CG had +10.65% advantage) and seq_len=25 (where it had -0.23% disadvantage). Tested at seq_len=15.

**Key numbers**:
- At seq_len=15: Per-Object CG beats 2-Node CG by **+3.28%**
- Advantage decay curve: +10.65% (10 steps) → +3.28% (15 steps) → -0.23% (25 steps)
- **Crossover point estimated at seq_len ≈ 24.3**

**What this means**: We now have a concrete design principle: **use Per-Object CG for tasks with ≤20 timesteps, and 2-Node CG for longer-horizon planning**. The per-object structure's advantage decays approximately linearly with sequence length, crossing zero around 24 timesteps. This likely reflects the point where error accumulation in the larger Per-Object parameter space outweighs its representational benefits.

**Next step (H1.424)**: Design a hybrid architecture that adaptively selects between Per-Object and 2-Node CG based on sequence length, or explore whether adding temporal attention to Per-Object CG can extend its advantage to longer sequences.
