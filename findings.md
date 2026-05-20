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

**Critical Insight**: Experience replay approaches are INCONCLUSIVE on multi-step tasks. The best configuration (Temporal Consistency alone at +0.41%) provides only marginal improvement, and combining replay with auxiliary losses actually degrades performance slightly (-0.35% for Replay+TC, -0.39% for EWC+TC). This suggests:

1. **Replay adds noise, not signal**: On this task, replaying past experiences doesn't provide additional useful gradients beyond what the main training loop already captures
2. **EWC over-regularizes**: The EWC penalty constrains the model too much, preventing it from adapting to the multi-step task distribution
3. **Temporal consistency is sufficient**: The simple temporal consistency auxiliary loss captures most of the benefit; adding replay mechanisms doesn't compound the gains
4. **Diminishing returns on regularization**: Multiple regularization techniques (replay + TC, EWC + TC) interfere with each other rather than complementing

**Why Replay Failed to Help**:
1. **Task simplicity**: The multi-step task may not have enough distributional diversity for replay to be useful
2. **Single-pass sufficiency**: The model may already see sufficient diversity in a single epoch of shuffled multi-complexity data
3. **Replay overhead**: The additional gradient steps from replay may disrupt the main training signal

**Comparison with Prior Results**:
- H1.470.1.1.34: Temporal consistency alone achieved +5.70% (larger model, more data)
- H1.470.1.1.35: Temporal consistency alone achieved +0.41% (smaller model, less data)
- The magnitude difference suggests the benefit of auxiliary losses scales with model/data size

**Next Steps**:
1. H1.470.1.1.36: Test whether auxiliary loss benefits scale with model size and data volume
2. H1.470.1.1.37: Test auxiliary losses on longer sequences (10+ timesteps) to see if benefits scale with sequence length
3. Investigate whether the cognitive graph architecture itself can be modified to better handle multi-step tasks (e.g., explicit sub-goal nodes)
4. Validate findings on real robot data

### H1.470.1.1.34: Auxiliary Loss Approaches for Multi-Step Tasks — Round 273 (SUPPORTED)

**Context**: Following H1.470.1.1.33's REFUTED result showing curriculum learning is harmful on multi-step tasks (-51.47% worse than baseline), this experiment tested whether auxiliary losses (sub-goal prediction, temporal consistency) could improve performance without the catastrophic forgetting caused by staged curriculum training.

**Hypothesis**: Auxiliary losses will improve multi-step task performance by providing additional gradient signals that encourage the model to learn intermediate representations, without staged training that causes forgetting.

**Configurations Tested**:
1. Baseline: Standard MSE loss on actions
2. Sub-goal Prediction: Auxiliary loss predicting intermediate states from hidden representation
3. Temporal Consistency: Loss enforcing smooth state transitions between steps
4. Combined: Sub-goal + temporal consistency with fixed weights
5. Weighted Auxiliary: Adaptive weighting based on loss magnitudes (uncertainty-based)

**Key Findings**:

| Configuration | Test Loss | vs Baseline |
|--------------|-----------|-------------|
| Baseline | 0.063061 | +0.00% |
| Sub-goal Prediction | 0.060041 | +4.79% |
| Temporal Consistency | 0.059469 | +5.70% |
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
| 1-step | 0.001350 | 0.024626 | -1724.04% |
| 2-step | 0.014963 | 0.021733 | -45.25% |
| 3-step | 0.017859 | 0.022442 | -25.66% |
| 4-step | 0.022503 | 0.030407 | -35.12% |

**Critical Insight**: Curriculum learning is REFUTED on multi-step tasks. Fixed curriculum performs -51.47% worse than baseline, and this degradation occurs across ALL complexity levels — even 1-step tasks suffer -1724% worse performance under fixed curriculum. This suggests catastrophic forgetting between curriculum stages: training on easy tasks first actually harms the model's ability to handle those same tasks later.

**Why This Failed**:
1. **Catastrophic forgetting**: Stage-by-stage training causes the model to overwrite knowledge from earlier stages when training on harder tasks
2. **Distribution shift**: Each curriculum stage trains on a different data distribution, preventing the model from learning a unified representation
3. **No benefit from staging**: The cognitive graph architecture already handles mixed-complexity data well; staging adds no value

**Key Takeaway**: Joint training on all complexity levels simultaneously is superior to staged curriculum learning for multi-step tasks. Auxiliary losses (particularly temporal consistency) provide a better alternative — they encourage structured learning without staged training.
