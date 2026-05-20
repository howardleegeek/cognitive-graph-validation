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

### H1.470: Error Accumulation in Unified Representations — Round 236 (ANALYSIS: Representation Bottleneck Hypothesis)

**Hypothesis**: CG's advantage decreases with task complexity because the fixed 512-dim unified representation becomes a bottleneck when encoding both current state and task history.

**Context**: H1.469 showed CG improvement drops from +8.07% (single-step) to +2.08% (3-step), a -5.99% difference. This contradicts the original H1 prediction.

**Analysis of H1.469 Data**:
- Single-step: baseline=0.011058, CG=0.010166, improvement=+8.07%
- 3-step: baseline=0.010440, CG=0.010224, improvement=+2.08%

**Critical observation**: Both architectures improve on multi-step vs single-step, but baseline improves MORE:
- Baseline: 0.011058 → 0.010440 (**5.59% better** on multi-step)
- CG: 0.010166 → 0.010224 (**0.57% worse** on multi-step)

**Mechanism Hypothesis**: The 512-dim unified space must encode both current state AND task history on multi-step tasks, creating an information bottleneck. Separated encoders (baseline) maintain independent representations, allowing the fusion layer to learn task-specific weighting.

**Sub-Hypothesis H1.470.1**: Increasing the unified representation dimension (512 → 1024+) will reduce the single-to-multi performance gap for CG.

**Falsification criteria**:
- REFUTED if: Larger representations don't improve multi-step performance relative to single-step
- REFUTED if: Baseline also improves proportionally (general capacity issue, not CG-specific)
- SUPPORTED if: CG multi-step improvement increases disproportionately with larger representations

**Status**: ANALYSIS COMPLETE — H1.470.1 experiment planned for next round

---

### H1.469: Multi-step Tasks Experiment — Round 235 (REFUTED: CG advantage decreases with complexity)

**Hypothesis**: CG advantage increases with task complexity. Prediction: 3-step tasks will show larger CG improvement than single-step.

**Method**: Compare CG vs baseline on single-step vs 3-step LIBERO-style tasks (400 train, 100 val).

**Results**:

| Task Type | Baseline Loss | CG Loss | CG Improvement |
|-----------|--------------|---------|----------------|
| Single-step | 0.011058 | 0.010166 | **+8.07%** |
| 3-step | 0.010440 | 0.010224 | **+2.08%** |

**Key Finding**: CG advantage drops by -5.99% from single-step to 3-step. Hypothesis REFUTED.

**Conclusion**: CG advantage does NOT increase with task complexity. The unified representation that helps on single-step tasks becomes a liability on multi-step tasks.

---

### H1.467: Dropout Rate Sweep — Round 233 (SUPPORTED: Optimal dropout at 40%)

**Hypothesis**: There exists an optimal dropout rate that maximizes CG's advantage over baseline. Prediction: 30-40% dropout will be optimal, balancing regularization with capacity.

**Context**: H1.466 showed Dropout CG (30%) generalizes to realistic robot data. This experiment finds the optimal dropout rate for deployment.

**Method**: Test CG with dropout rates [0%, 10%, 20%, 30%, 40%, 50%, 60%] against baseline on synthetic LIBERO-style data (400 train, 100 val demos).

**Results**:

| Dropout Rate | Loss | vs Baseline | CG Wins |
|--------------|------|-------------|---------|
| **Baseline** | 0.010846 | — | — |
| 0% | 0.011323 | -4.39% | ✗ |
| 10% | 0.010159 | +6.33% | ✓ |
| 20% | 0.010641 | +1.89% | ✓ |
| 30% | 0.010086 | +7.01% | ✓ |
| **40%** | **0.009724** | **+10.34%** | **✓** |
| 50% | 0.009726 | +10.32% | ✓ |
| 60% | 0.009748 | +10.12% | ✓ |

**Key Findings**:
1. **Optimal dropout at 40%**: Peak improvement of +10.34% over baseline
2. **No dropout = worse than baseline**: 0% dropout CG loses to baseline by 4.39%
3. **Plateau effect**: 40-60% dropout all perform similarly well (+10.1% to +10.3%)
4. **Prediction confirmed**: Optimal rate (40%) falls within predicted 30-40% range

**Conclusion**: SUPPORTED — Optimal dropout rate is 40%, confirming the prediction that moderate regularization balances capacity and robustness. The plateau from 40-60% suggests the architecture is tolerant to over-regularization.

---

### H1.466: Dropout CG on Real Robot Data — Round 232 (SUPPORTED: Dropout CG generalizes to realistic conditions)

**Hypothesis**: Dropout CG (30%) architectural robustness generalizes to realistic deployment conditions.

**Context**: H1.465 showed Drop
