# H1.470: Error Accumulation in Unified Representations — Deep Analysis

## Round 236

### Context
H1.469 (Round 235) showed CG advantage drops from 8.07% (single-step) to 2.08% (3-step), a -5.99% difference. This contradicts the original H1 prediction that CG advantage increases with task complexity.

### Deep Analysis of H1.469 Data

From H1.469 results:
- **Single-step**: baseline=0.011058, CG=0.010166, improvement=+8.07%
- **3-step**: baseline=0.010440, CG=0.010224, improvement=+2.08%

**Critical observation**: Both architectures improve on multi-step vs single-step, but baseline improves MORE:
- Baseline: 0.011058 → 0.010440 (**5.59% better** on multi-step)
- CG: 0.010166 → 0.010224 (**0.57% worse** on multi-step)

This means:
1. The multi-step task is actually *easier* for both architectures (lower absolute loss)
2. But CG loses its relative advantage — it can't capitalize on the easier task as much as baseline
3. The unified representation that helps on single-step becomes a liability on multi-step

### Mechanism Hypothesis: Representation Bottleneck

The CG architecture maps both physical (8-dim) and semantic (32-dim) inputs into a 512-dim unified space (144 physical + 368 semantic). On single-step tasks, this unified space allows cross-modal grounding. But on multi-step tasks:

1. **Information bottleneck**: The 512-dim space must encode both the current state AND the task history
2. **Cross-modal interference**: Physical errors contaminate semantic representations and vice versa
3. **Baseline advantage**: Separated encoders maintain independent representations, allowing the fusion layer to learn task-specific weighting

### Concrete Sub-Hypothesis: H1.470.1

**Hypothesis**: CG's performance degradation on multi-step tasks is caused by the fixed 512-dim representation being insufficient to encode both current state and task history simultaneously.

**Prediction**: Increasing the unified representation dimension (e.g., 512 → 1024) will reduce the performance gap between single-step and multi-step tasks for CG.

**Falsification criteria**:
- REFUTED if: Increasing representation dimension doesn't improve multi-step performance relative to single-step
- REFUTED if: Baseline also improves proportionally (suggesting it's a general capacity issue, not CG-specific)
- SUPPORTED if: CG multi-step improvement increases disproportionately with larger representation

### Test Plan for H1.470.1

1. Run CG with representation dimensions [256, 512, 768, 1024, 2048]
2. Test each on single-step and 3-step tasks
3. Measure: improvement vs baseline at each dimension, for each task type
4. Key metric: Does the single-to-multi performance gap shrink with larger representations?

### Status: ANALYSIS COMPLETE — H1.470.1 experiment planned for next round
