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

### H1.470.1.1.29: Phase-Aware + Ensemble Disagreement for Mixed Tasks — Round 268 (REFUTED)

**Context**: Building on H1.470.1.1.28's success with phase-aware training (+99.05% to +99.82% on hierarchical tasks), this experiment tested whether combining phase-aware training with ensemble disagreement could handle mixed tasks that have BOTH hierarchical structure AND sensor noise.

**Hypothesis**: A hybrid approach (phase-aware for hierarchical parts + ensemble disagreement for noise) would outperform either approach alone on mixed tasks.

**Configurations Tested**:
1. Baseline: Standard training without any weighting
2. Phase-aware: Upweights phase transitions
3. Ensemble disagreement: Downweights high-disagreement (noisy) samples
4. Hybrid: Combines both phase-aware and disagreement weighting

**Key Findings**:

1. **Test Results (3 phases, noise=0.1, seq_len=18)**:
   | Strategy | Test Loss | Improvement |
   |----------|-----------|-------------|
   | Baseline | 10.07 | +0.00% |
   | Phase-aware | 10.74 | -6.67% |
   | Ensemble disagreement | 9.66 | +4.03% |
   | Hybrid | 10.90 | -8.24% |

2. **Test Results (4 phases, noise=0.15, seq_len=28)**:
   | Strategy | Test Loss | Improvement |
   |----------|-----------|-------------|
   | Baseline | 19.09 | +0.00% |
   | Phase-aware | 18.87 | +1.13% |
   | Ensemble disagreement | 17.20 | +9.91% |
   | Hybrid | 17.65 | +7.51% |

3. **Test Results (5 phases, noise=0.2, seq_len=40)**:
   | Strategy | Test Loss | Improvement |
   |----------|-----------|-------------|
   | Baseline | 26.67 | +0.00% |
   | Phase-aware | 27.16 | -1.83% |
   | Ensemble disagreement | 29.62 | -11.06% |
   | Hybrid | 26.74 | -0.28% |

4. **Summary**:
   - Phase-aware: -2.46% average (worse than baseline)
   - Ensemble disagreement: +0.96% average (marginal improvement)
   - Hybrid: -0.34% average (slightly worse)

**Conclusion**: REFUTED - Neither phase-aware training nor hybrid approaches provide consistent improvement on mixed tasks. The benefits of phase-aware training are specific to tasks with clear hierarchical structure, not general-purpose.

**Key Insight**: Phase-aware training works when there are clear, learnable phase transitions (hierarchical tasks). On mixed/noisy tasks, the "phase transitions" are less distinct and weighting them actually hurts performance. This confirms that H1.470.1.1.28's success was task-specific to hierarchical manipulation.

---

### H1.470.1.1.28: Phase-Aware Training for Hierarchical Tasks — Round 267 (SUPPORTED)

**Context**: H1.470.1.1.27 showed ensemble disagreement fails on hierarchical tasks (-1.35% to -1.59%). The key insight was that ensemble disagreement downweights phase transitions, which are "uncertain" but highly informative for learning task structure. This experiment tests whether phase-aware training that UPWEIGHTS phase transitions improves performance.

**Hypothesis**: Phase-aware training that explicitly incorporates phase labels and transition information will improve performance on hierarchical multi-step tasks, unlike noise-aware loss that downweights transitions.

**Configurations Tested**:
1. Baseline: Standard training without phase awareness
2. Oracle phase: Ground truth phase labels + transition information (upper bound)
3. Detected phase: Automatic phase transition detection from velocity changes

**Key Findings**:

1. **Test Loss Comparison (3-Phase Tasks)**:
   | Strategy | Test Loss | Improvement |
   |----------|-----------|-------------|
   | Baseline | 1.06e-05 | +0.00% |
   | Oracle phase | 4.25e-08 | +99.60% |
   | Detected phase | 5.46e-08 | +99.48% |

2. **Test Loss Comparison (4-Phase Tasks)**:
   | Strategy | Test Loss | Improvement |
   |----------|-----------|-------------|
   | Baseline | 1.17e-05 | +0.00% |
   | Oracle phase | 1.24e-09 | +99.99% |
   | Detected phase | 2.13e-07 | +98.18% |

3. **Test Loss Comparison (5-Phase Tasks)**:
   | Strategy | Test Loss | Improvement |
   |----------|-----------|-------------|
   | Baseline | 1.03e-05 | +0.00% |
   | Oracle phase | 1.18e-08 | +99.89% |
   | Detected phase | 5.29e-08 | +99.49% |

4. **Critical Result**: **Phase-aware training dramatically improves hierarchical task learning.**
