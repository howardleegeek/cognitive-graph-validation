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

### H1.470.1.1.39: Task-Dependent Regularization Scaling — Round 278 (INCONCLUSIVE)

**Context**: H1.470.1.1.38 showed over-regularization occurs at different capacity thresholds for different architectures. This experiment tests whether regularization should scale with task complexity (trajectory variance, temporal dependencies, state space coverage).

**Hypothesis**: Regularization weight should increase with task complexity to prevent overfitting on more complex tasks.

**Configurations Tested**:
- Task complexities: low (linear reach), medium (waypoint navigation), high (pick-and-place)
- Model sizes: h=32, 64, 128
- Regularization weights: 0.0, 0.01, 0.1
- Model types: Simple GRU, Cognitive Graph
- Total configurations: 54

**Key Findings**:

1. **CRITICAL: L2 Regularization HURTS Performance Across ALL Task Complexities**
   | Task Complexity | Optimal Reg Weight | Overfitting Detected |
   |-----------------|-------------------|---------------------|
   | Low | 0.0 | 0/6 configs |
   | Medium | 0.0 | 0/6 configs |
   | High | 0.0 | 6/6 configs |

2. **Train-Val Gap Analysis Reveals Underfitting**:
   - Low complexity: All models show NEGATIVE train-val gap (underfitting)
   - Medium complexity: All models show NEGATIVE train-val gap (underfitting)
   - High complexity: All models show POSITIVE train-val gap (overfitting)

3. **Cognitive Graph Consistently Outperforms Simple GRU**:
   | Task | GRU h=64 | CG h=64 | CG Improvement |
   |------|----------|---------|----------------|
   | Low | 0.0339 | 0.0219 | +35.5% |
   | Medium | 0.0522 | 0.0342 | +34.5% |
   | High | 0.1120 | 0.0634 | +43.4% |

4. **Correlation Analysis**:
   - Correlation between task complexity and regularization benefit: 0.000
   - No evidence that regularization should scale with task complexity

**Conclusion**: The hypothesis is NOT supported. Regularization does not help at any task complexity level. The key insight is that overfitting only emerges at high task complexity, but even then, zero regularization is optimal. This suggests the models are capacity-limited rather than overfitting-prone.

**Implications**:
- Focus on data augmentation for high-complexity tasks, not regularization
- Consider smaller model capacity for low-complexity tasks to reduce underfitting
- Task-aware model selection (capacity scaling with complexity) may be more effective than task-aware regularization

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

## Summary of Hypotheses Status

| Hypothesis | Status | Key Finding |
|-----------|--------|-------------|
| H1: Cognitive Graph Sample Efficiency | SUPPORTED | +25.6% improvement with real robot data |
| H2: Statistical Significance | INCONCLUSIVE | 1.7% difference, needs more data |
| H3: Attention vs Concatenation | REFUTED | Concatenation wins for simple tasks |
| H4: Dimension Allocation | CLOSE | 25% optimal vs 28% hypothesis |
| H1.470.1.1.36: Scaling Auxiliary Loss | REFUTED | Small models benefit, large models hurt |
| H1.470.1.1.37: Adaptive Regularization | REFUTED | Fixed regularization wins |
| H1.470.1.1.38: Architecture-Dependent Over-Reg | INCONCLUSIVE | Both architectures over-regularize at different thresholds |
| H1.470.1.1.39: Task-Dependent Regularization | INCONCLUSIVE | No regularization needed; overfitting only at high complexity |

## Research Direction

The regularization experiments (H1.470.1.1.36-39) reveal a consistent pattern:
1. L2 regularization generally hurts performance
2. Models tend to underfit rather than overfit
3. Overfitting only emerges at high task complexity
4. Cognitive Graph consistently outperforms Simple GRU

**Next Steps**:
- H1.470.1.1.40: Investigate task-aware model capacity scaling
- Explore data augmentation for high-complexity tasks
- Consider early stopping as alternative to regularization