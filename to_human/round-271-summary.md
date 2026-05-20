# Round 271 Summary: Adaptive Curriculum Scheduling Test

**Experiment**: H1.470.1.1.32 - Test adaptive curriculum scheduling based on learning progress

**Result**: REFUTED (-17.16% worse than fixed curriculum)

**Key Findings**:
1. Adaptive curriculum scheduling performed worse than fixed curriculum scheduling
2. Surprisingly, the baseline (no curriculum at all) performed best with test loss of 0.149382
3. Fixed curriculum had test loss of 0.301490, adaptive curriculum had 0.353221
4. This contradicts the previous round's finding that curriculum learning provides +81.09% improvement

**Insights**:
- Simple training on all data may be sufficient for smooth trajectory tasks
- Learning-progress-based adaptive scheduling needs more sophisticated metrics
- The benefits of curriculum learning appear to be highly task-dependent
- There may be inconsistencies in the synthetic data generation between experiments

**Next Step**: H1.470.1.1.33 - Test curriculum learning on more complex multi-step tasks to better understand when curriculum approaches are beneficial.