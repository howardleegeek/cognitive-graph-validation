# Round 266 Summary: Ensemble Disagreement on Complex Hierarchical Tasks

**Experiment**: H1.470.1.1.27 - Tested ensemble disagreement noise estimation on complex hierarchical multi-step tasks with 4-5 phases.

**Result**: REFUTED. Ensemble disagreement fails on hierarchical tasks regardless of complexity:
- 3-phase tasks: -4.05% improvement (worse than baseline)
- 4-phase tasks: -1.35% improvement (still worse)
- 5-phase tasks: -1.59% improvement (still worse)

**Key Insight**: Increasing task complexity does NOT help ensemble disagreement. The technique works well for tasks with genuine noise/uncertainty (real robot data: +15.24%) but fails on structured hierarchical tasks where phase transitions are deterministic and learnable. Oracle noise also fails on hierarchical tasks (-0.49% to +0.11%), suggesting noise-aware loss is fundamentally the wrong approach for structured tasks.

**Next Action**: Investigate phase-aware training methods for hierarchical tasks, or explore alternative uncertainty estimation approaches that can distinguish between "uncertain but informative" samples (phase transitions) and "uncertain and noisy" samples.