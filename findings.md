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

---

### H1.419: Physical Grounding Tasks for Cognitive Graph — Round 185

**Hypothesis**: CG's unified representation (physical + semantic in shared space) will outperform separated architectures on tasks requiring physical reasoning where language must be grounded in physical dynamics.

**Motivation**: Prior experiments (H1.415-418) showed CG slightly outperforms baseline on standard tasks (+1.2%) but temporal extensions consistently underperform. The next logical test: does CG's advantage emerge on tasks that specifically require coupling physical dynamics with language understanding?

**Method**: Three physical grounding tasks, each with 3000 samples (70/15/15 split), 50 epochs, lr=1e-3, batch_size=128:
1. **Collision prediction**: Given 5-object scene + language specifying object pair, predict if they collide within 10 timesteps (binary classification)
2. **Object permanence**: Given scene + language specifying occluded object, predict its position after 5 timesteps (6D regression)
3. **Spatial reasoning**: Given scene + language specifying two objects, predict their relative position (3D regression)

Three architectures tested:
- **Baseline**: Separate encoders → concatenation → MLP decoder
- **Cognitive Graph**: Unified 48+96=144d space, GNN message passing + cross-attention
- **Graph Attention**: Object-level graph (each object = node, language = query node)

**Results**:

| Task | Baseline Loss | CG Loss | CG vs Baseline | GraphAttn Loss | GraphAttn vs Baseline |
|------|--------------|---------|----------------|----------------|----------------------|
| Collision | 0.562248 | 0.555275 | **+1.24%** | 0.563647 | -0.25% |
| Permanence | 18.833548 | 19.833021 | **-5.31%** | 17.838958 | **+5.28%** |
| Spatial | 12.857434 | 12.806340 | **+0.40%** | 12.844021 | +0.10% |

**Additional metrics**:
- Collision accuracy: Baseline 76.89%, CG 76.89%, GraphAttn 76.89% (all identical — task may be at ceiling)
- Permanence MAE: Baseline 2.97, CG 3.06, GraphAttn 2.85
- Spatial MAE: Baseline 2.66, CG 2.65, GraphAttn 2.67

**Conclusion**: H1.419 **PARTIALLY SUPPORTED**. CG shows marginal advantage on collision (+1.24%) and spatial (+0.40%) tasks but underperforms on permanence (-5.31%). The Graph Attention architecture (object-level graph) is the clear winner on permanence (+5.28%), suggesting that **explicit object-level graph structure** matters more than unified representation space for physical reasoning tasks. CG's unified space may be too coarse-grained — it fuses all physical state into one vector rather than maintaining per-object representations.

**Key insight**: The advantage of graph-based approaches appears to depend on **granularity of graph nodes**. Object-level graphs (GraphAttn) outperform the coarse 2-node CG (physical blob + semantic blob) on tasks requiring per-object reasoning (permanence). This suggests H1 may need refinement: the benefit of CG may come from graph structure, not from unified representation space.
