# Round 197 Summary - H1.431: Testing CG on Relational Tasks

**Experiment**: Tested whether Cognitive Graph (CG) would outperform Baseline MLP on tasks with explicit relational structure (collision avoidance, stacking, pushing) where graph inductive bias should help.

**Key Result**: CG performed **22.5-32.8% worse** than Baseline MLP across all three relational tasks, decisively refuting the hypothesis. Baseline MLP achieved MSE of 0.005362 (collision), 0.002405 (stacking), and 0.019028 (pushing), while CG achieved 0.006859, 0.002946, and 0.025266 respectively.

**Implication**: The CG architecture's underperformance is not limited to simple synthetic tasks — it persists even on tasks explicitly designed to benefit from relational reasoning. This raises fundamental questions about whether the unified graph representation provides any sample efficiency advantage, as CG fails to outperform even simple MLP baselines on tasks where its inductive bias should be most beneficial.

**Next Step**: H1.432 will analyze failure modes to understand why CG underperforms despite its theoretical advantages, investigating architectural limitations vs optimization issues.