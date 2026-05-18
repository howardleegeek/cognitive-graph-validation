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

### H1.418: Transformer-based Temporal Cognitive Graph — Round 184

**Hypothesis**: Transformer architecture will better capture temporal dependencies in multi-step robotic tasks compared to GRU-based Temp-CG variants.

**Previous context**: All Temp-CG variants (v1 self-recurrent, v2 proper GRU, v3 curriculum) underperformed baseline by 36-63%. Transformer attention may handle long-range dependencies better than GRU.

**Method**:
1. Replace GRU recurrence with transformer encoder layers
2. Test with varying sequence lengths (1, 2, 3, 5, 10 steps)
3. Compare against baseline MLP and original CG
4. Scaled-down unified space (48+96=144d), n_train=1000, n_val=250, n_test=250, epochs=50

**Results**:
| Model | Test Loss | vs Baseline |
|-------|-----------|-------------|
| Baseline (MLP) | 0.308591 | — |
| Cognitive Graph | 0.305024 | +1.2% |
| Transformer-CG | 0.310434 | -0.6% |

**Per-sequence-length breakdown**:
| Steps | Baseline | CG | Trans-CG |
|-------|----------|----|----------|
| 1 | 0.5303 | 0.5754 | 0.4518 |
| 2 | 0.3671 | 0.4938 | 0.3998 |
| 3 | 0.2586 | 0.4241 | 0.3423 |
| 5 | 0.2250 | 0.4602 | 0.3318 |
| 10 | 0.0730 | 0.0384 | 0.0698 |

**Conclusion**: H1.418 INCONCLUSIVE. Transformer-CG (-0.6%) slightly underperforms baseline. Original CG (+1.2%) slightly outperforms baseline. Neither temporal extension shows strong advantage. The 10-step case shows CG (0.0384) outperforming both baseline (0.0730) and Trans-CG (0.0698), suggesting CG may have some long-horizon capability but is not consistently better.

---

### H1.415-417: Temporal CG — Extended Training, Proper GRU, Curriculum Learning — Round 183

**Hypotheses**:
- H1.415: Training Temp-CG for 200+ epochs will recover performance (training difficulty hypothesis)
- H1.416: Proper GRU with separate input/hidden states will fix the self-recurrent flaw
- H1.417: Curriculum learning (short→long sequences) will stabilize Temp-CG training

**Method**:
1. Three Temp-CG variants tested against baseline and original CG
2. v1: Original self-recurrent GRU (H1.414 architecture, 100 epochs)
3. v2: Proper GRU with state_dynamics network computing meaningful input (H1.416)
4. v3: Same as v2 but with curriculum: 30ep@3steps → 30ep@5steps → 40ep@10steps (H1.417)
5. Scaled-down unified space (48+96=144d) for tractable CPU training
6. n_train=1000, n_val=250, n_test=250, max_steps=10, epochs=100, lr=1e-3 (Temp-CG), lr=3e-4 (baseline/CG)

**Results**:
| Model | Final Val Loss | Best Val Loss | Best Epoch | vs Baseline |
|-------|---------------|---------------|------------|-------------|
| Baseline (MLP) | 0.024565 | 0.024562 | 91 | — |
| Cognitive Graph | 0.040236 | 0.039302 | 41 | -63.79% |
| Temp-CG v1 (self-recurrent, 100ep) | 0.033412 | 0.032596 | 60 | -36.01% |
| Temp-CG v2 (proper GRU, 100ep) | 0.036512 | 0.035929 | 58 | -48.63% |
| Temp-CG v3 (curriculum, 100ep) | 0.039312 | 0.038454 | 49 | -60.03% |

**Per-sequence-length breakdown**:
| Steps | Baseline | CG | Temp-CG v1 | Temp-CG v2 | Temp-CG v3 |
|-------|----------|----|------------|------------|------------|
| 1 | 0.0238 | 0.0656 | 0.0606 | 0.0529 | 0.0595 |
| 2 | 0.0204 | 0.0468 | 0.0492 | 0.0579 | 0.0499 |
| 3 | 0.0213 | 0.0458 | 0.0368 | 0.0351 | 0.0409 |
| 5 | 0.0242 | 0.0314 | 0.0219 | 0.0225 | 0.0307 |
| 10 | 0.0324 | 0.0872 | 0.0758 | 0.0815 | 0.0790 |
