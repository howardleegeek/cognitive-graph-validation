# Round 192 Summary

**H1.426: Per-Object CG with Explicit Relational Edges**

Tested whether adding explicit spatial relational edges (above, below, beside, near) improves Per-Object CG on spatial relation tasks. **Result: NOT_SUPPORTED** — adding explicit relational edges HURTS performance (+22.91% worse). Standard Per-Object CG achieves -5.79% vs baseline, outperforming all architectures including 2-Node CG (+25.34% worse). This reveals an important pattern: Per-Object CG excels at tasks where individual object states matter (spatial relations) but struggles with multi-stage manipulation (H1.425). The 2-Node architecture shows the opposite pattern. This suggests task-dependent architecture selection may be warranted.
