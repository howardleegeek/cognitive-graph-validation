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

### H1.384: Decomposition Pattern Analysis (Round 155)

**Hypothesis**: CG's implicit decomposition through cross-modal attention creates more coherent task representations than explicit hierarchical subgoal structure.

**Prediction**: CG's intermediate representations will show better clustering by task phase, smoother transitions, and higher mutual information with ground-truth subgoals.

**Results**:

| Model | Val MSE | Phase Silhouette | Subgoal Silhouette | ARI (Phase) | ARI (Subgoal) |
|-------|---------|------------------|--------------------| ------------|---------------|
| Baseline (LSTM) | 0.004896 | **0.0465** | **0.1590** | **0.4455** | **0.3595** |
| Hierarchical Planner | **0.004588** | -0.0043 | 0.1245 | 0.2351 | 0.0893 |
| Cognitive Graph | 0.005071 | 0.0115 | 0.1192 | 0.3016 | 0.1775 |

**Status: ⚠️ REFUTED** — Key observations:

1. **Hierarchical planner wins on MSE**: -10.53% vs CG, -6.27% vs baseline
2. **Baseline shows BEST decomposition quality**: Highest silhouette scores and ARI for both phase and subgoal clustering
3. **CG underperforms on all decomposition metrics**: Lower phase clustering (0.0115 vs 0.0465), lower subgoal clustering (0.1192 vs 0.1590)
4. **Hierarchical planner shows NEGATIVE phase clustering**: Silhouette -0.0043 indicates poor phase separation

**Implications**: The hypothesis that CG's implicit decomposition is superior is refuted. Surprisingly, the simple baseline achieves the best decomposition quality despite worse MSE than hierarchical. This suggests:
- Explicit subgoal structure (hierarchical) improves prediction but not representation quality
- CG's unified representation does NOT automatically yield better task decomposition
- The relationship between representation quality and prediction accuracy is non-trivial

**Critical Finding**: CG's advantage in prior experiments may be task-specific, not due to superior decomposition.

---

### H1.383: Implicit vs Explicit Task Decomposition (Round 154)

**Hypothesis**: CG's implicit task decomposition through cross-modal attention outperforms explicit subgoal structure because the unified representation learns more flexible decompositions.

**Prediction**: CG without explicit subgoal supervision will match or exceed hierarchical planner with explicit subgoals.

**Results**:

| Model | MSE | Improvement vs Baseline |
|-------|-----|------------------------|
| Flat Baseline | 0.016397 | — |
| Hierarchical Planner | 0.014181 | +13.51% |
| Cognitive Graph (no supervision) | 0.014025 | **+14.47%** |
| Cognitive Graph (with supervision) | 0.014032 | +14.42% |

**Status: ✅ SUPPORTED** — CG's implicit decomposition (+14.47%) slightly outperforms hierarchical (+13.51%). Explicit supervision provides minimal benefit (+0.05% difference).

---

### H1.382: Curriculum Asymmetry Analysis (Round 153)

**Hypothesis**: The hierarchical planner's explicit subgoal decomposition structure naturally benefits from curriculum learning because it has modular structure.

**Results**:

| Model | 4-step MSE | Improvement vs Baseline | Curriculum Benefit |
|-------|-----------|------------------------|-------------------|
| Flat Baseline (LSTM) | 0.412749 | — | — |
| Hierarchical Planner (Direct) | 0.343173 | +16.86% | — |
| Hierarchical Planner (Curriculum) | 0.305071 | +26.09% | +9.23% |
| Cognitive Graph (Direct) | 0.272718 | +33.93% | — |
| Cognitive Graph (Curriculum, no supervision) | 0.243814 | **+40.93%** | +7.00% |
| Cognitive Graph (Curriculum + supervision) | 0.269338 | +34.75% | -6.18% (harmful!) |

**Status: ⚠️ PARTIALLY REFUTED** — Subgoal supervision HURTS CG (-6.18%). CG curriculum without supervision achieves best results (+40.93%).

---

## Summary Status

| Hypothesis | Status | Key Evidence |
|------------|--------|--------------|
| H1: CG > Separated | SUPPORTED | +25.6% improvement with real robot data |
| H2: Attention vs Concat | INCONCLUSIVE | 1.7% difference |
| H3: Attention for long sequences | REFUTED | Concatenation wins for simple tasks |
| H4: 25% optimal threshold | CLOSE | 25% optimal vs 28% hypothesis |
| H1.382: Curriculum asymmetry | PARTIALLY REFUTED | Supervision hurts CG |
| H1.383: Implicit decomposition | SUPPORTED | +14.47% vs +13.51% |
| H1.384: Decomposition quality | REFUTED | Baseline has best decomposition metrics |

## Open Questions

1. Why does baseline have better decomposition quality but worse prediction than hierarchical?
2. Why does CG underperform on decomposition metrics despite competitive MSE?
3. Is CG's advantage specific to certain task types (e.g., longer horizons, more complex language)?
4. What is the relationship between representation quality and prediction accuracy?

## Next Actions

1. **H1.385**: Test CG on longer sequences (20+ timesteps) to see if decomposition advantage emerges
2. **H1.386**: Analyze attention patterns in CG to understand what it learns instead of phase decomposition
3. **H1.387**: Test with more complex multi-step tasks (3+ subgoals) where explicit decomposition may help