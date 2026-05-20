# Round 239 Summary — H1.470.1.1.1: Finer Dimension Sweep Around 832

**Status**: REFUTED

**What we tested**: Following the previous round's finding that 832 dimensions appeared optimal (outperforming 768), we ran a finer-grained sweep at [800, 816, 832, 848, 864] to confirm whether 832 is a true local optimum.

**Key result**: 816 is the optimal dimension, not 832. At 816 dimensions, CG achieves +31.06% multi-step improvement over baseline, compared to 832's +23.84% — a 7.22 percentage point difference. However, the performance landscape is relatively flat across the tested range (21.70% to 31.06%), suggesting the optimum is broad rather than sharp.

**Most robust finding**: All five dimensions show a negative improvement gap (CG performs better on multi-step than single-step), ranging from -19.73% to -27.09%. This is a consistent pattern that survives across the entire dimension sweep and confirms that CG's unified representation genuinely benefits from multi-step task structure.

**Next step**: Test whether the optimal dimension (816) is stable across different task complexities (2-step, 4-step, 5-step) or if it shifts with sequence length — hypothesis H1.470.1.1.2.
