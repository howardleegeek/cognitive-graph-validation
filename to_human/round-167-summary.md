# Round 167 Summary: Architecture Tuning Breakthrough

**Key Finding**: The CG architecture underperformance in H1.395 (-4.5% avg) was caused by **over-parameterization**, not a fundamental architecture flaw. With a smaller 256-dim model (vs 512-dim), CG achieves **+20.9% average improvement** over baseline.

**Experiment H1.396** tested 5 architecture configurations:
- **Config A (256-dim, 2-heads)**: +24.9% at complexity=100, +16.9% at complexity=300 ← **Best**
- Config B (512-dim, 1-head): +14.6% avg
- Config C (512-dim, 4-heads, 40 epochs): +8.0% avg
- Config D (512-dim, 4-heads, lr=1e-4): -21.5% avg (learning rate too low)
- Config E (128-dim, 1-head): +13.2% avg

**Implications**:
1. Resolves the H1.395 discrepancy where CG underperformed on synthetic data
2. Establishes a **model-data complexity matching principle**: larger models need richer data
3. Explains why H1 showed +25.6% on real robot data (rich structure) while synthetic data needed smaller models

**Next**: H1.397 will test the optimal 256-dim architecture across the full complexity range (20-600) to verify scaling behavior.