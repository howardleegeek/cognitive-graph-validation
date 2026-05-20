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

### H1.470.1.1.36: Scaling Auxiliary Loss Benefits — Round 275 (REFUTED)

**Context**: Following H1.470.1.1.34's SUPPORTED result showing temporal consistency auxiliary loss improves multi-step performance (+5.70%), and H1.470.1.1.35's INCONCLUSIVE result showing experience replay doesn't compound gains, this experiment tested whether auxiliary loss benefits scale with model size and data volume.

**Hypothesis**: Auxiliary loss benefits (particularly temporal consistency) will scale positively with both model size (more parameters = better regularization benefit) and data volume (more data = more stable auxiliary signal).

**Configurations Tested**:
- Model sizes: small (hidden=32), medium (hidden=64), large (hidden=128)
- Data volumes: 500, 1000, 2000 samples
- Loss types: baseline MSE vs temporal consistency

**Key Findings**:

| Model Size | Data | Baseline Loss | TC Loss | Improvement |
|------------|------|--------------|---------|-------------|
| Small (32) | 500 | 0.021598 | 0.019577 | **+9.36%** |
| Small (32) | 1000 | 0.012351 | 0.011982 | **+2.99%** |
| Small (32) | 2000 | 0.007866 | 0.007614 | **+3.20%** |
| Medium (64) | 500 | 0.018394 | 0.021100 | -14.71% |
| Medium (64) | 1000 | 0.013152 | 0.012639 | +3.91% |
| Medium (64) | 2000 | 0.008978 | 0.009042 | -0.72% |
| Large (128) | 500 | 0.018770 | 0.019007 | -1.26% |
| Large (128) | 1000 | 0.012169 | 0.013802 | -13.42% |
| Large (128) | 2000 | 0.009430 | 0.009700 | -2.87% |

**Average Improvement by Model Size**:
- Small (32): **+5.18%**
- Medium (64): -3.84%
- Large (128): -5.85%

**Critical Insight**: The hypothesis is REFUTED. Temporal consistency regularization helps small models but **hurts large models**. This is an over-regularization effect: larger models have more capacity to learn the task directly, and auxiliary losses constrain them unnecessarily. Small models benefit from the inductive bias that temporal consistency provides.

**Why This Matters**:
1. **Regularization-capacity tradeoff**: Auxiliary losses are beneficial when model capacity is limited, harmful when capacity is sufficient
2. **No data scaling effect**: More data doesn't amplify auxiliary loss benefits (improvement stays flat across 500→2000 samples)
3. **Practical implication**: Use temporal consistency loss only for small models (hidden_dim ≤ 32), rely on data volume for larger models

**Recommendations**:
- R1: Apply temporal consistency loss only to under-capacity models
- R2: For larger models, increase data volume rather than adding regularization
- R3: Investigate adaptive regularization that scales with model capacity

### H1.470.1.1.35: Experience Replay + Auxiliary Losses for Multi-Step Tasks — Round 274 (INCONCLUSIVE)

**Context**: Following H1.470.1.1.34's SUPPORTED result showing temporal consistency auxiliary loss improves multi-step performance (+5.70%), and H1.470.1.1.33's REFUTED result showing curriculum learning causes catastrophic forgetting (-51.47%), this experiment tested whether experience replay (uniform and prioritized) combined with auxiliary losses could further improve performance by providing diverse gradient signals across task complexities.

**Hypothesis**: Experience replay combined with auxiliary losses will further improve multi-step task performance by preventing catastrophic forgetting and providing diverse gradient signals across task complexities.

**Configurations Tested**:
1. Baseline: Standard MSE loss
2. Temporal Consistency: Auxiliary loss for smooth transitions (replicating H1.470.1.1.34)
3. Experience Replay: Uniform replay buffer with MSE
4. Replay + Temporal Consistency: Combined approach
5. Prioritized Replay + TC: Weight harder samples more
6. EWC + Temporal Consistency: Elastic Weight Consolidation to prevent forgetting

**Key Findings**:

| Configuration | Test Loss | vs Baseline |
|--------------|-----------|-------------|
| Baseline | 0.022540 | +0.00% |
| Temporal Consistency | 0.022448 | +0.41% |
| Experience Replay | 0.022538 | +0.01% |
| Replay + Temporal Consistency | 0.022619 | -0.35% |
| Prioritized Replay + TC | 0.022507 | +0.15% |
| EWC + Temporal Consistency | 0.022629 | -0.39% |

**Critical Insight**: Experience replay approaches are INCONCLUSIVE on multi-step tasks. The best configuration (Temporal Consistency alone at +0.41%) provides only marginal improvement, and combining replay with auxiliary losses actually degrades performance. Replay adds noise rather than signal for this task type.

**Why Experience Replay Failed**:
1. **No distribution shift**: Multi-step tasks don't have the staged training that causes catastrophic forgetting
2. **Replay noise**: Random sampling from replay buffer introduces variance without benefit
3. **EWC over-regularization**: Elastic Weight Consolidation prevents adaptation to multi-step distribution

### H1.470.1.1.34: Auxiliary Losses for Multi-Step Tasks — Round 273 (SUPPORTED)

**Context**: Following H1.470.1.1.33's REFUTED result showing curriculum learning causes catastrophic forgetting (-51.47%), this experiment tested whether auxiliary losses could improve multi-step task performance by providing structured gradient signals without staged training.

**Hypothesis**: Auxiliary losses (temporal consistency, subgoal prediction) will improve multi-step task performance by encouraging the model to learn intermediate representations and smooth state transitions.

**Configurations Tested**:
1. Baseline: Standard MSE loss
2. Subgoal Prediction: Predict intermediate states
3. Temporal Consistency: Enforce smooth transitions
4. Combined: Subgoal + Temporal Consistency
5. Weighted Auxiliary: Higher weight on temporal consistency

**Key Findings**:

| Configuration | Test Loss | vs Baseline |
|--------------|-----------|-------------|
| Baseline | 0.063061 | +0.00% |
| Subgoal Prediction | 0.060041 | +4.79% |
| Temporal Consistency | 0.059469 | **+5.70%** |
| Combined | 0.060781 | +3.62% |
| Weighted Auxiliary | 0.059764 | +5.23% |

**Per-Complexity Analysis (Baseline vs Temporal Consistency)**:

| Complexity | Baseline Loss | Consistency Loss | Improvement |
|------------|--------------|-----------------|-------------|
| 1-step | 0.096000 | 0.096485 | -0.51% |
| 2-step | 0.074471 | 0.067777 | +8.99% |
| 3-step | 0.063353 | 0.060992 | +3.73% |
| 4-step | 0.059960 | 0.059127 | +1.39% |

**Critical Insight**: Auxiliary losses are SUPPORTED on multi-step tasks, with temporal consistency providing the largest improvement (+5.70%). The benefit is strongest on 2-step tasks (+8.99%) and diminishes with complexity, suggesting auxiliary losses help most at intermediate complexity levels where the model has enough capacity to benefit from structured gradients but isn't overwhelmed by task complexity.

**Why Auxiliary Losses Succeed Where Curriculum Failed**:
1. **No staged training**: All data is seen simultaneously, avoiding catastrophic forgetting
2. **Implicit structure learning**: Auxiliary losses encourage the model to learn intermediate representations without explicit stage boundaries
3. **Gradient regularization**: Auxiliary losses act as regularizers that prevent overfitting to any single complexity level
4. **Temporal consistency is key**: The best-performing auxiliary loss enforces smooth state transitions, which directly addresses the sequential nature of multi-step tasks

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
| 1-step | 0.001350 | 0.024626 | -1724. |