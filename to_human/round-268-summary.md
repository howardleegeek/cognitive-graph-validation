# Round 268 Summary

**Experiment**: H1.470.1.1.29 - Phase-Aware + Ensemble Disagreement for Mixed Tasks

**Result**: REFUTED

**Key Findings**:
Tested whether combining phase-aware training (from H1.470.1.1.28's success) with ensemble disagreement could handle mixed tasks with BOTH hierarchical structure AND sensor noise. Results showed neither approach provides consistent improvement:
- Phase-aware: -2.46% average (worse than baseline)
- Ensemble disagreement: +0.96% average (marginal)
- Hybrid: -0.34% average (slightly worse)

**Insight**: The dramatic benefits of phase-aware training (+99% in H1.470.1.1.28) are specific to tasks with clear hierarchical structure. On mixed/noisy tasks, phase transitions are less distinct and weighting them actually hurts performance. This confirms the task-specific nature of these techniques.

**Next**: Test phase-aware training on real robot LIBERO data with known phase boundaries.
