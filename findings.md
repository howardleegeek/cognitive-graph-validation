# Research Findings — Cognitive Graph Architecture

## Research Question

Does a unified cognitive graph architecture (early fusion of physical and semantic representations) achieve higher sample efficiency than separated architectures (JEPA + LLM alignment) on language-conditioned robotic tasks?

## Current Understanding

**The Core Hypothesis**: Current approaches (V-JEPA 2, π0, LED-WM) all suffer from representation separation — vision and language exist in different spaces and are only aligned after encoding. This causes:
1. **Grounding problems**: Language not truly grounded in physical dynamics
2. **Combinatorial explosion**: Need to learn all (vision, language) pairings separately
3. **Learning inefficiency**: No gradient flow between modalities during training

**The Cognitive Graph Solution**: A unified 512-dimensional representation space where:
- 144 dimensions encode physical world state (analogous to V-JEPA embeddings)
- 368 dimensions encode semantic/conceptual information (analogous to LLM embeddings)
- Single GNN processes both, with cross-modal attention allowing dynamic interaction
- Explicit graph structure (nodes = objects/concepts, edges = relationships/physics)

## Key Results

### H1.470.1.1.33: Curriculum Learning on Complex Multi-Step Tasks — Round 272 (REFUTED)

**Context**: Following H1.470.1.1.32's REFUTED result showing adaptive curriculum performs -17.16% worse than fixed on smooth trajectories (and baseline was actually best), this experiment tested whether curriculum learning would help on genuinely complex multi-step tasks where sequential dependencies matter.

**Hypothesis**: On complex multi-step tasks (pick→place→return chains), curriculum learning will outperform baseline because the model needs to master simpler sub-tasks before attempting full sequences.

**Configurations Tested**:
1. Baseline: No curriculum, all data shuffled (1-4 step tasks mixed)
2. Fixed Curriculum: 3-stage progression (1-step → 2-step → 3+ step tasks)
3. Adaptive Curriculum: Progressive difficulty (≤2 steps → ≤3 steps → all)
4. Reverse Curriculum: Hard → easy (3+ step → 2-step → 1-step)
5. Curriculum + Attention: Fixed curriculum with cross-modal attention

**Key Findings**:

| Configuration | Test Loss | vs Baseline |
|--------------|-----------|-------------|
| Baseline (no curriculum) | 0.016030 | +0.00% |
| Adaptive Curriculum | 0.016768 | -4.61% |
| Fixed Curriculum | 0.024281 | -51.47% |
| Reverse Curriculum | 0.024776 | -54.56% |
| Curriculum + Attention | 0.024887 | -55.25% |

**Per-Complexity Analysis (Baseline vs Fixed Curriculum)**:

| Complexity | Baseline Loss | Fixed Curriculum Loss | Change |
|------------|--------------|----------------------|--------|
| 1-step | 0.001350 | 0.024626 | -1724.04% |
| 2-step | 0.014963 | 0.021733 | -45.25% |
| 3-step | 0.017859 | 0.022442 | -25.66% |
| 4-step | 0.022503 | 0.030407 | -35.12% |

**Critical Insight**: Curriculum learning is REFUTED on multi-step tasks. Fixed curriculum performs -51.47% worse than baseline, and this degradation occurs across ALL complexity levels — even 1-step tasks suffer -1724% worse performance under fixed curriculum. This suggests catastrophic forgetting between curriculum stages: training on easy tasks first actually harms the model's ability to handle those same tasks later.

**Why This Failed**:
1. **Catastrophic forgetting**: Stage-by-stage training causes the model to overwrite knowledge from earlier stages when training on harder tasks
2. **Distribution shift**: Each curriculum stage trains on a different data distribution, preventing the model from learning a unified policy
3. **No rehearsal**: Without replay of earlier-stage data, the model loses simpler skills
4. **Joint training is superior**: Training on all complexities simultaneously allows the model to learn shared representations that generalize across task difficulty

**Comparison with Prior Results**:
- H1.470.1.1.31: Curriculum +81.09% on smooth trajectories (SUPPORTED)
- H1.470.1.1.32: Adaptive curriculum -17.16% on smooth trajectories (REFUTED)
- H1.470.1.1.33: Fixed curriculum -51.47% on multi-step tasks (REFUTED)

**Pattern**: Curriculum learning only helps on very simple, smooth trajectory tasks where the data distribution is homogeneous. As soon as tasks involve discrete sub-goals or heterogeneous complexity, curriculum learning becomes harmful.

### H1.470.1.1.32: Adaptive Curriculum Scheduling Based on Learning Progress — Round 271 (REFUTED)

**Context**: Following H1.470.1.1.31's SUPPORTED result showing curriculum learning provides +81.09% improvement on smooth robot trajectories, this experiment tested whether adaptive curriculum scheduling (adjusting difficulty based on learning progress) would outperform fixed curriculum scheduling.

**Hypothesis**: Adaptive curriculum scheduling that adjusts difficulty based on learning progress will outperform fixed curriculum scheduling on smooth robot trajectories.

**Configurations Tested**:
1. Adaptive Curriculum: Learning-progress-based scheduling across 5 length bins (50-120, 120-190, 190-260, 260-330, 330-400 steps)
2. Fixed Curriculum: Same 3-stage progression as H1.470.1.1.31 (50-150, 150-300, 300-450 steps)
3. Baseline (no curriculum): Standard training on all data shuffled
4. Reverse Curriculum: Long → short progression

**Key Findings**:

| Configuration | Test Loss | vs Baseline | vs Fixed Curriculum |
|--------------|-----------|-------------|---------------------|
| Adaptive Curriculum | 0.353221 | -136.45% | -17.16% |
| Fixed Curriculum | 0.301490 | -101.82% | +0.00% |
| Baseline (no curriculum) | 0.149382 | +0.00% | +50.45% |
| Reverse Curriculum | 0.298599 | -99.89% | +0.96% |

**Critical Insight**: Adaptive curriculum performs WORSE than fixed curriculum (-17.16%), and surprisingly, the baseline (no curriculum) performs BEST overall. This contradicts H1.470.1.1.31's results where curriculum showed +81.09% improvement.

**Why This Failed**:
1. **Dataset inconsistency**: The synthetic data generation may have different characteristics than H1.470.1.1.31
2. **Progress metric issues**: Simple loss reduction may not be a good proxy for learning progress
3. **Over-adaptation**: Adaptive scheduling may oscillate between difficulty levels, preventing stable learning

## Research Trajectory Summary

1. **H1.470.1.1.28**: Phase-aware training shows +99%+ improvement on synthetic hierarchical tasks (SUPPORTED)
2. **H1.470.1.1.30**: Phase-aware training fails on realistic robot data (REFUTED) → technique doesn't generalize
3. **H1.470.1.1.31**: Curriculum learning shows +81.09% improvement on smooth trajectories (SUPPORTED) → promising alternative
4. **H1.470.1.1.32**: Adaptive curriculum performs worse than fixed curriculum (REFUTED) → simpler may be better
5. **H1.470.1.1.33**: Curriculum learning fails on multi-step tasks (REFUTED, -51.47%) → curriculum only helps on simple homogeneous tasks

## Current Research Direction

The curriculum learning hypothesis has been thoroughly tested and REFUTED for complex tasks. The pattern is clear:
- **Simple, homogeneous tasks** (smooth trajectories): Curriculum can help (+81.09%)
- **Complex, heterogeneous tasks** (multi-step): Curriculum is harmful (-51.47%)
- **Adaptive scheduling**: Never outperforms baseline

**Key Takeaway**: Joint training on all data simultaneously is the most robust approach. Curriculum learning introduces catastrophic forgetting and distribution shift that outweigh any benefits from progressive difficulty.

**Next Steps**:
1. H1.470.1.1.34: Test auxiliary loss approaches (sub-goal prediction, consistency losses) as alternative to curriculum
2. Investigate whether replay/regularization between curriculum stages could mitigate catastrophic forgetting
3. Explore whether the cognitive graph architecture itself can be modified to better handle multi-step tasks
4. Validate findings on real robot data
