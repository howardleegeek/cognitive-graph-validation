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

### H1.470.1.1.30: Phase-Aware Training on LIBERO-style Data — Round 269 (REFUTED)

**Context**: Following H1.470.1.1.28's dramatic success with phase-aware training (+99.05% to +99.82% on synthetic hierarchical tasks) and H1.470.1.1.29's failure on mixed/noisy tasks, this experiment tested whether phase-aware training would help on LIBERO-style robot manipulation data with clear phase structure (approach → grasp → lift → transport → place).

**Hypothesis**: Phase-aware training would significantly improve learning on robot manipulation tasks with clear phase structure, similar to the synthetic hierarchical task results.

**Configurations Tested**:
1. Baseline: Standard MSE training
2. Oracle phase-aware: Ground truth phase labels, weight=3.0
3. Detected phase-aware: Predicted phases with auxiliary loss, weight=3.0
4. Oracle phase weight 2.0, 5.0, 10.0: Different weighting strengths

**Key Findings**:

| Configuration | Test Loss | Improvement vs Baseline |
|--------------|-----------|------------------------|
| Baseline | 0.000146 | +0.00% |
| Oracle phase-aware (w=3.0) | 0.000214 | -47.15% |
| Detected phase-aware (w=3.0) | 0.000462 | -217.15% |
| Oracle phase weight 2.0 | 0.000207 | -42.42% |
| Oracle phase weight 5.0 | 0.000212 | -45.88% |
| Oracle phase weight 10.0 | 0.000226 | -54.88% |

**Critical Insight**: ALL phase-aware configurations performed WORSE than baseline. The best phase-aware config (weight=2.0) was still 42.42% worse than baseline.

**Why This Failed**:
1. **Task complexity mismatch**: LIBERO-style manipulation has smooth, continuous trajectories where phase transitions are less critical than synthetic hierarchical tasks with discrete phase boundaries
2. **Loss weighting interference**: Upweighting phase transitions distorts the overall loss landscape, causing the model to overfit to transition points at the expense of overall trajectory accuracy
3. **Phase detection overhead**: The detected phase-aware model had to learn phase prediction as an auxiliary task, adding complexity without benefit
4. **Different learning dynamics**: The synthetic hierarchical tasks in H1.470.1.1.28 had sharp phase boundaries with distinct dynamics per phase, while LIBERO manipulation has overlapping dynamics across phases

**Conclusion**: REFUTED - Phase-aware training does NOT transfer to LIBERO-style robot manipulation data. The dramatic improvements seen in H1.470.1.1.28 are specific to synthetic tasks with sharp phase boundaries and do not generalize to realistic robot manipulation trajectories.

**Implications**:
- Phase-aware training is NOT a general technique for robot learning
- The success in H1.470.1.1.28 was an artifact of synthetic task design
- Need to explore alternative approaches for real robot data

---

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

**Conclusion**: REFUTED - Neither phase-aware training nor hybrid approaches provide consistent improvement on mixed tasks. The benefits of phase-aware training are specific to tasks with clear hierarchical structure and no noise.

---

### H1.470.1.1.28: Phase-Aware Training for Hierarchical Tasks — Round 267 (SUPPORTED)

**Context**: Testing whether phase-aware training (upweighting loss at phase transitions) improves learning on hierarchical multi-step tasks.

**Hypothesis**: Phase transitions are critical learning moments; upweighting them should improve sample efficiency.

**Key Findings**:

| Configuration | Test Loss | Improvement |
|--------------|-----------|-------------|
| Baseline (3 phases) | 1.06e-05 | +0.00% |
| Oracle phase (3 phases) | 4.25e-08 | +99.60% |
| Detected phase (3 phases) | 5.46e-08 | +99.48% |
| Baseline (4 phases) | 1.17e-05 | +0.00% |
| Oracle phase (4 phases) | 1.24e-09 | +99.99% |
| Detected phase (4 phases) | 2.13e-07 | +98.18% |
| Baseline (5 phases) | 1.03e-05 | +0.00% |
| Oracle phase (5 phases) | 1.18e-08 | +99.89% |
| Detected phase (5 phases) | 5.29e-08 | +99.49% |

**Average Improvements**:
- Oracle phase-aware: +99.82%
- Detected phase-aware: +99.05%

**Conclusion**: SUPPORTED - Phase-aware training dramatically improves hierarchical task learning. Automatic phase detection from velocity changes works nearly as well as oracle labels.

**IMPORTANT CAVEAT (from H1.470.1.1.30)**: This result does NOT generalize to realistic robot manipulation data. The synthetic tasks had sharp phase boundaries that don't exist in real robot trajectories.

---

## Summary of Hypotheses

| Hypothesis | Status | Key Evidence |
|------------|--------|---------------|
| H1: Unified CG improves sample efficiency | SUPPORTED | +25.6% on real robot data |
| H2: Attention helps long sequences | INCONCLUSIVE | 1.7% difference |
| H3: Attention beats concatenation | REFUTED | Concatenation wins for simple tasks |
| H4: 25% optimal dimension allocation | CLOSE | 25% optimal vs 28% hypothesis |
| H1.470.1.1.28: Phase-aware for hierarchical | SUPPORTED* | +99.05% to +99.82% (*synthetic only) |
| H1.470.1.1.29: Phase-aware + ensemble for mixed | REFUTED | -2.46% average |
| H1.470.1.1.30: Phase-aware for LIBERO data | REFUTED | -42.42% to -217.15% |

## Next Steps

1. **Investigate why phase-aware fails on real data**: The synthetic tasks had sharp phase boundaries; real robot trajectories are smooth
2. **Alternative approaches for real robot data**: Consider curriculum learning, data augmentation, or different loss weighting strategies
3. **Re-examine H1.470.1.1.28 results**: The dramatic improvements may be artifacts of synthetic task design