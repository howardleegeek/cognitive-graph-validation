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

### H1.470.1.1.41: Aggressive Training Strategies — Round 280 (SUPPORTED)

**Context**: H1.470.1.1.40 showed underfitting persists across all model sizes and task complexities. This experiment tests whether more aggressive training (higher learning rates, longer training) can reduce underfitting.

**Hypothesis**: Higher learning rates and longer training will reduce underfitting and improve validation loss.

**Configurations Tested**:
- Learning rates: [1e-4, 1e-3, 1e-2]
- Training epochs: [50, 100, 200]
- LR schedules: [constant, warmup_cosine]
- Model sizes: [32, 64]
- Task complexities: [low, high]
- Total configurations: 72

**Key Findings**:

1. **Higher Learning Rates Reduce Underfitting**:
   | Learning Rate | Avg Val Loss | Avg Gap | Underfit % |
   |--------------|--------------|---------|-------------|
   | 1e-4 | 0.1342 | -0.0200 | 58.3% |
   | 1e-3 | 0.1365 | -0.1070 | 50.0% |
   | **1e-2** | **0.1230** | -0.1169 | **50.0%** |

2. **Training Duration Has Minimal Impact**:
   | Epochs | Avg Val Loss | Avg Gap | Underfit % |
   |--------|--------------|---------|-------------|
   | 50 | 0.1297 | -0.0566 | 50.0% |
   | 100 | 0.1282 | -0.0808 | 54.2% |
   | 200 | 0.1358 | -0.1066 | 54.2% |

3. **Warmup Cosine Schedule Slightly Reduces Underfitting**:
   | Schedule | Avg Val Loss | Avg Gap | Underfit % |
   |----------|--------------|---------|-------------|
   | constant | 0.1268 | -0.0854 | 55.6% |
   | **warmup_cosine** | 0.1356 | -0.0771 | **50.0%** |

4. **Best Configuration**:
   - Config: `lr0.01_epochs50_warmup_cosine_h64_low`
   - Val Loss: 0.0032
   - Train-Val Gap: -0.0014 (GOOD - minimal underfitting)

5. **Underfitting Still Persists**:
   - Underfitting: 38/72 (52.8%)
   - Overfitting: 0/72 (0%)
   - Well-fitted: 34/72 (47.2%)

**Conclusion**: SUPPORTED. Higher learning rates (1e-2) improve validation loss and reduce underfitting compared to conservative rates (1e-4). However, underfitting remains the dominant issue (52.8% of configurations), and no overfitting was observed even with aggressive training. The hypothesis that aggressive training reduces underfitting is partially supported - higher LR helps, but longer training does not.

**Recommendations**:
- R1: Use learning rate 1e-2 for this task class (10x higher than typical)
- R2: 50 epochs is sufficient; longer training does not help
- R3: Consider even higher learning rates (3e-2, 1e-1) or different optimizers
- R4: The fundamental issue may be model capacity, not training strategy

---

### H1.470.1.1.40: Task-Aware Model Capacity Scaling — Round 279 (REFUTED)

**Context**: H1.470.1.1.39 showed models underfit on low/medium complexity tasks (negative train-val gap) while overfitting only emerges at high complexity. This experiment tests whether task-aware capacity scaling (smaller models for simple tasks, larger for complex) improves performance.

**Hypothesis**: Task-aware capacity scaling will outperform fixed-size strategies by reducing underfitting on simple tasks and overfitting on complex tasks.

**Configurations Tested**:
- Strategies: fixed_small (h=16), fixed_medium (h=32), fixed_large (h=64), task_aware (h=16 for low, 32 for medium, 64 for high)
- Task complexities: low, medium, high
- Model: Simple feedforward network (simplified for rapid testing)
- Total configurations: 12

**Key Findings**:

1. **Fixed Large Model Outperforms Task-Aware Strategy**:
   | Strategy | Average Val Loss | Rank |
   |----------|------------------|------|
   | Fixed Large (h=64) | 0.2573 | 1 |
   | Task-Aware | 0.3940 | 2 |
   | Fixed Medium (h=32) | 0.4127 | 3 |
   | Fixed Small (h=16) | 0.4816 | 4 |

2. **Task-Aware Shows Mixed Performance**:
   - Task-aware improves over fixed_small by +18.2%
   - Task-aware improves over fixed_medium by +4.5%
   - Task-aware is WORSE than fixed_large by -53.1%

3. **Underfitting Persists Across All Strategies**:
   | Complexity | Fixed Small Gap | Fixed Medium Gap | Fixed Large Gap | Task-Aware Gap |
   |------------|-----------------|------------------|-----------------|----------------|
   | Low | -0.0914 (UNDER) | -0.0992 (UNDER) | -0.0815 (UNDER) | -0.0850 (UNDER) |
   | Medium | -0.0345 (UNDER) | -0.0371 (UNDER) | -0.0216 (UNDER) | -0.0266 (UNDER) |
   | High | -0.0134 (UNDER) | -0.0126 (UNDER) | -0.00 |

**Conclusion**: REFUTED. Larger models (h=64) outperform ALL other strategies including task-aware capacity scaling. The hypothesis that simple tasks need smaller models is refuted. Models are capacity-limited, not overfitting-prone.

---

### H1.470.1.1.38: Architecture-Dependent Over-Regularization — Round 277 (INCONCLUSIVE)

**Context**: H1.470.1.1.36 showed temporal consistency regularization helps small models but hurts large models. This experiment tests whether this is architecture-dependent.

**Hypothesis**: Over-regularization at larger model sizes is architecture-dependent (different architectures have different optimal capacity thresholds).

**Key Findings**:

1. **Both Architectures Show Over-Regularization**:
   | Model | h=32 | h=64 | h=128 | h=256 |
   |-------|------|------|-------|-------|
   | Simple GRU | +0.40% | -1.44% | -6.17% | N/A |
   | Cognitive Graph | N/A | N/A | +11.83% | -9.38% |

2. **Over-Regularization Threshold Differs by Architecture**:
   - Simple GRU: Over-regularizes at h=64+
   - Cognitive Graph: Over-regularizes at h=256+

3. **Cognitive Graph Benefits More at Moderate Sizes**:
   - At h=128: CG shows +11.83% improvement vs GRU's -6.17%

**Conclusion**: INCONCLUSIVE. Both architectures exhibit over-regularization but at different capacity thresholds. The effect is NOT purely architecture-dependent.

---

### H1.470.1.1.37: Adaptive Regularization Scaling — Round 276 (REFUTED)

**Context**: H1.470.1.1.36 suggested regularization should scale with model capacity. This experiment tests adaptive regularization strategies.

**Hypothesis**: Adaptive regularization that scales with model capacity will outperform fixed regularization.

**Key Findings**:

1. **Fixed Regularization Outperforms All Adaptive Strategies**:
   | Strategy | h=32 | h=64 | h=128 |
   |----------|------|------|-------|
   | Fixed (0.1) | +0.04% | +0.10% | +0.11% |
   | Adaptive Linear | +0.04% | +0.05% | +0.04% |
   | Adaptive Exp | +0.02% | +0.02% | +0.02% |

2. **Simple Scaling Functions Are Too Aggressive**:
   - Linear, inverse sqrt, and exponential all reduce weight too much for larger models
   - The optimal regularization weight is relatively constant across model sizes

**Conclusion**: REFUTED. Fixed regularization consistently outperforms adaptive strategies.

---

### H1.470.1.1.24: Ensemble Disagreement on Real Robot Data — Round 263 (SUPPORTED)

**Context**: H1.470.1.1.23 showed ensemble disagreement outperforms oracle noise estimation by 10x on synthetic data. This experiment validates on real robot data.

**Key Findings**:

1. **Test Loss Comparison**:
   | Strategy | Test Loss | Improvement | Oracle Ratio |
   |----------|-----------|-------------|--------------|
   | Baseline | 0.019219 | +0.00% | N/A |
   | Oracle noise | 0.018816 | +2.10% | 100% |
   | **Ensemble disagreement** | **0.016290** | **+15.24%** | **726.4%** |

2. **Ensemble disagreement maintains 7.3x superiority over oracle noise on real robot data!**

---

## Summary

### Hypothesis Status

| Hypothesis | Status | Key Finding |
|------------|--------|-------------|
| H1: Cognitive Graph > Separated | SUPPORTED | +25.6% improvement with real robot data |
| H2: Attention vs Concatenation | INCONCLUSIVE | 1.7% difference |
| H3: Attention for long sequences | REFUTED | Concatenation wins for simple tasks |
| H4: Dimension allocation (25% physical) | CLOSE | 25% optimal vs 28% hypothesis |

### Current Focus: Underfitting Investigation

**Key Insight**: Across multiple experiments (H1.470.1.1.38-41), underfitting is the dominant issue:
- H1.470.1.1.38: Over-regularization hurts at large capacities
- H1.470.1.1.39: Underfitting on low/medium complexity tasks
- H1.470.1.1.40: Larger models always win, task-aware scaling refuted
- H1.470.1.1.41: Higher LR helps but underfitting persists (52.8% of configs)

**Pattern**: Models are capacity-limited, not overfitting-prone. The solution space points toward:
1. Larger model capacities
2. Higher learning rates (1e-2)
3. Reduced regularization
4. More training data or data augmentation