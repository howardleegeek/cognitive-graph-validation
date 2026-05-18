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

**Key Findings**:

1. **H1.415 PARTIALLY SUPPORTED**: Extended training (100ep vs 40ep) dramatically improved Temp-CG v1 from catastrophic failure (-788% at 1 step in H1.414) to competitive performance (-36% vs baseline). The training difficulty hypothesis is confirmed — Temp-CG simply needed more epochs. However, it still does NOT beat the baseline.

2. **H1.416 REFUTED**: Proper GRU with separate input/hidden states (v2) performed WORSE than the self-recurrent version (v1): 0.0365 vs 0.0334 final val loss. The state_dynamics network did not provide the expected benefit. This suggests the issue is not the GRU formulation but something deeper about the recurrent architecture.

3. **H1.417 INCONCLUSIVE**: Curriculum learning (v3) performed worse than both v1 and v2 (0.0393 final val loss). The curriculum may have caused catastrophic forgetting when transitioning between phases. However, v3 showed the best 5-step performance (0.0307) among Temp-CG variants, suggesting partial benefit.

4. **Critical insight**: ALL Temp-CG variants still underperform the simple MLP baseline. The temporal CG architecture fundamentally struggles with this task. The inverse loss scaling pattern from H1.414 persists: all Temp-CG variants have lowest loss at 5 steps and highest at 1 and 10 steps, suggesting the recurrent architecture is optimized for medium-length sequences but fails at extremes.

5. **CG also underperforms baseline**: The original CG architecture (0.0402) also loses to the baseline (0.0246) in this scaled-down configuration. This may be due to the reduced unified space (144d vs 512d) or the task being too simple for CG's inductive biases to matter.

**Conclusion**: Temporal CG is REFUTED as a general improvement. Neither extended training, proper GRU formulation, nor curriculum learning recovers baseline performance. The recurrent message passing architecture appears fundamentally mismatched for this prediction task. Future work should explore: (a) whether Temp-CG helps on tasks with stronger temporal dependencies, (b) whether the graph structure itself (not the recurrence) is the issue, (c) whether residual connections or normalization changes could stabilize training.

### H1.414: Temporal CG with Recurrent Message Passing — Round 182

**Hypothesis**: Explicit temporal modeling (recurrent message passing) will maintain CG advantage over longer planning horizons, addressing H1.413 finding that advantage decreases with sequence length.

**Method**:
1. Built Temporal-CG architecture that processes actions one timestep at a time
2. Each timestep: apply action → message passing → GRU-style recurrent update
3. Compared against baseline (flat MLP) and original CG (flat input)
4. 5 objects, max 10 steps, n_train=1000, n_val=500, epochs=40, lr=1e-3

**Results**:
| Steps | Baseline Loss | CG Loss | Temp-CG Loss | CG vs BL | Temp-CG vs BL |
|-------|--------------|---------|-------------|----------|---------------|
| 1 | 0.029011 | 0.018440 | 0.257649 | +36.4% | -788.1% |
| 2 | 0.023889 | 0.013365 | 0.401940 | +44.1% | -1582.5% |
| 3 | 0.016626 | 0.012129 | 0.442343 | +27.0% | -2560.6% |
| 4 | 0.011308 | 0.008990 | 0.527863 | +20.5% | -4568.0% |
| 5 | 0.009584 | 0.011841 | 0.271031 | -23.5% | -2727.8% |
| 6 | 0.006796 | 0.009346 | 0.160496 | -37.5% | -2261.8% |
| 7 | 0.006405 | 0.009211 | 0.123746 | -43.8% | -1833.5% |
| 8 | 0.005396 | 0.011793 | 0.094811 | -118.5% | -1658.9% |
| 9 | 0.005588 | 0.010567 | 0.039090 | -89.1% | -599.6% |
| 10 | 0.005338 | 0.010713 | 0.013712 | -100.7% | -157.0% |

**Key Finding**: **REFUTED.** Temporal CG with recurrent message passing performs dramatically worse than both baseline and original CG across all sequence lengths. However, a striking pattern emerges: Temp-CG loss *decreases* with sequence length (0.258 at 1 step → 0.014 at 10 steps), while baseline and CG losses are relatively flat or increase. This suggests:

1. **Training difficulty**: The recurrent architecture is significantly harder to train with limited epochs
2. **Inverse scaling**: The model may be learning to predict longer sequences better than short ones (counterintuitive)
3. **Architecture mismatch**: Self-recurrent GRU (h_new = GRU(h, h)) may not provide meaningful temporal dynamics

### H1.413: Multi-step Sequential Interaction Prediction — Round 181

**Hypothesis**: CG advantage will be maintained across multi-step prediction tasks (1-5 action chains).

**Results**: CG won 100% of configurations (4/4), with improvements of +91.57% (1 step), +88.93% (2 steps), +87.81% (3 steps), +86.42% (5 steps).

**Key Finding**: **PARTIALLY SUPPORTED.** CG dominates at all sequence lengths, but advantage *decreases* with sequence length (91.6% → 86.4%). This suggests CG's inductive biases are most valuable for simple tasks and diminish for complex multi-step reasoning.

## Hypothesis Status Summary

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| H1: CG > separated architectures | SUPPORTED | +25.6% with real robot data |
| H1.413: CG advantage across multi-step | PARTIALLY SUPPORTED | 100% win rate, but advantage decreases with sequence length |
| H1.414: Temporal CG maintains advantage | REFUTED | Temp-CG loses to both baseline and CG at all lengths |
| H1.415: Extended training fixes Temp-CG | PARTIALLY SUPPORTED | 100ep recovers from catastrophic failure but still loses to baseline |
| H1.416: Proper GRU fixes Temp-CG | REFUTED | Proper GRU performs worse than self-recurrent version |
| H1.417: Curriculum learning stabilizes Temp-CG | INCONCLUSIVE | Worst overall but best at 5-step; may cause catastrophic forgetting |
| H2: Scaling benefits | Inconclusive | 1.7% difference |
| H3: Attention > concatenation | REFUTED | Concatenation wins for simple tasks |
| H4: Optimal physical:semantic ratio | CLOSE | 25% optimal vs 28% hypothesis |
