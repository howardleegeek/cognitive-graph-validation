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

4. **Critical Result**: **Phase-aware training dramatically improves hierarchical task learning.** Average oracle improvement: 99.82%, average detected improvement: 99.05%. This is a complete reversal from ensemble disagreement which degraded performance.

5. **Why Phase-Aware Training Works**:
   - Phase transitions are highly informative for learning task structure
   - Explicit phase encoding provides strong learning signal
   - Automatic detection from velocity changes works nearly as well as oracle labels
   - The key is to RECOGNIZE phase structure, not to downweight "uncertain" samples

6. **Comparison with Ensemble Disagreement**:
   | Method | Hierarchical Tasks | Real Robot Data |
   |--------|-------------------|-----------------|
   | Ensemble disagreement | -1.35% to -1.59% | +15.24% |
   | Phase-aware training | +99.05% to +99.82% | Not tested |

**Conclusion**: SUPPORTED. Phase-aware training is highly effective for hierarchical multi-step tasks. The key insight is that phase transitions should be explicitly modeled and used as learning signal, not treated as noise to be downweighted. This suggests a fundamental principle: **structured uncertainty (phase transitions) should be modeled, while random uncertainty (sensor noise) should be downweighted**.

---

### H1.470.1.1.27: Ensemble Disagreement on Complex Hierarchical Tasks — Round 266 (REFUTED)

**Context**: H1.470.1.1.26 showed ensemble disagreement performed -4.05% worse than baseline on 3-phase hierarchical tasks. This experiment tests whether more complex hierarchical structures (4-5 phases) provide better signal for ensemble disagreement to capture useful uncertainty.

**Hypothesis**: Ensemble disagreement will show improved performance on more complex hierarchical tasks (4-5 phases) because more phase transitions create more uncertainty points and longer sequences allow better calibration.

**Configurations Tested**:
1. Baseline: Standard training without noise-aware loss
2. Oracle noise: Ground truth noise levels (upper bound)
3. Ensemble disagreement: 5-model ensemble variance as noise estimate

**Key Findings**:

1. **Test Loss Comparison (4-Phase Tasks)**:
   | Strategy | Test Loss | Improvement | Oracle Ratio |
   |----------|-----------|-------------|--------------|
   | Baseline | 0.009572 | +0.00% | N/A |
   | Oracle noise | 0.009619 | -0.49% | 100% |
   | Ensemble disagreement | 0.009700 | -1.35% | 272.3% |

2. **Test Loss Comparison (5-Phase Tasks)**:
   | Strategy | Test Loss | Improvement | Oracle Ratio |
   |----------|-----------|-------------|--------------|
   | Baseline | 0.009002 | +0.00% | N/A |
   | Oracle noise | 0.008992 | +0.11% | 100% |
   | Ensemble disagreement | 0.009145 | -1.59% | -1435.5% |

3. **Critical Result**: **Ensemble disagreement fails on hierarchical multi-step tasks regardless of complexity.** The improvement went from -4.05% (3-phase) to -1.35% (4-phase) to -1.59% (5-phase). Increasing complexity does NOT help ensemble disagreement.

4. **Why Ensemble Disagreement Fails on Hierarchical Tasks**:
   - Hierarchical tasks have structured phase transitions, not random noise
   - Ensemble disagreement treats phase transitions as "uncertain" samples
   - Downweighting phase transitions removes important learning signal
   - Oracle noise also fails (-0.49% to +0.11%), suggesting noise-aware loss is fundamentally wrong approach for hierarchical tasks
   - The task structure is deterministic and learnable, not noisy

**Conclusion**: REFUTED. Ensemble disagreement noise estimation is NOT suitable for hierarchical multi-step tasks. The technique works well for tasks with genuine noise/uncertainty (real robot data: +15.24%) but fails on structured hierarchical tasks where phase transitions are deterministic and learnable.

---

## Hypothesis Status Summary

| Hypothesis | Status | Key Evidence |
|------------|--------|--------------|
| H1: Cognitive Graph improves sample efficiency | SUPPORTED | +25.6% improvement with real robot data |
| H2: Early fusion improves grounding | INCONCLUSIVE | 1.7% difference, needs more testing |
| H3: Attention enables long-horizon planning | REFUTED | Concatenation wins for simple tasks; attention needs task structure |
| H4: Dimension allocation (144/368) is optimal | CLOSE | 25% optimal vs 28% hypothesis |
| H1.470: Ensemble disagreement helps noisy data | PARTIALLY SUPPORTED | Works for real robot (+15.24%), fails for hierarchical tasks |
| H1.470.1.1.28: Phase-aware training for hierarchical tasks | SUPPORTED | +99.05% to +99.82% improvement |

## Key Principles Discovered

1. **Structured vs Random Uncertainty**: Phase transitions (structured) should be modeled; sensor noise (random) should be downweighted.
2. **Task Structure is Key**: Attention mechanisms need explicit task structure to work on long sequences.
3. **Ensemble Disagreement Domain**: Works for tasks with genuine noise/uncertainty, not for deterministic structured tasks.
4. **Phase-Aware Training**: Explicit phase encoding dramatically improves hierarchical task learning.

## Next Research Directions

1. Test phase-aware training on real robot data (does it help there too?)
2. Combine phase-aware training with ensemble disagreement for mixed tasks
3. Develop automatic phase detection for real-world manipulation tasks
4. Investigate phase-aware attention mechanisms for longer sequences