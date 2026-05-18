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

1. **Training difficulty**: The recurrent architecture is significantly harder to train with 40 epochs. The GRU cell may need much more training time to converge.
2. **Inverse scaling**: Temp-CG's loss pattern (high at short sequences, low at long) is the inverse of what we'd expect. This may indicate the model is learning to "average" over timesteps rather than properly propagate state.
3. **Architecture mismatch**: The self-recurrent GRU (h_new = GRU(h, h)) may not provide meaningful temporal dynamics. A proper input-to-hidden recurrent connection is needed.

**Analysis**: The hypothesis that recurrent message passing would maintain CG advantage is refuted under current training conditions. However, the inverse loss scaling pattern is intriguing and warrants investigation. The Temp-CG may simply need more training epochs, a different recurrent formulation, or curriculum learning (train on short sequences first, then extend).

**Next steps**: 
- H1.415: Train Temp-CG for 200+ epochs to check if it converges
- H1.416: Try proper GRU with separate input/hidden states
- H1.417: Curriculum learning approach (short→long sequences)

### H1.413: Multi-Step Sequential Interaction Prediction — Round 181 (Supplementary)

**Hypothesis**: CG advantage compounds over longer planning horizons (more sequential actions).

**Method**:
1. Built on H1.412 physics simulator with contact-based multi-object dynamics
2. Task: Given initial positions + sequence of N push actions, predict final positions
3. Tested sequence lengths: 1, 2, 3, 5 actions
4. 5 objects, n_train=1000, n_val=500, epochs=30

**Results**:
| Steps | Baseline Loss | CG Loss | Improvement | CG Wins |
|-------|--------------|---------|-------------|---------|
| 1 | 0.003926 | 0.000331 | +91.57% | ✓ |
| 2 | 0.005681 | 0.000629 | +88.93% | ✓ |
| 3 | 0.006328 | 0.000772 | +87.81% | ✓ |
| 5 | 0.007481 | 0.001233 | +83.52% | ✓ |

**Key Finding**: **PARTIALLY SUPPORTED.** CG maintains strong advantage across all sequence lengths (83-92%), but the relative improvement *decreases* slightly with more steps. Both models degrade with longer sequences, but CG degrades proportionally more (loss increases 3.7x vs baseline's 1.9x). 

**Analysis**: This suggests that while CG's relational reasoning provides a strong baseline advantage, error compounding over multiple steps affects both architectures. The CG's advantage is most pronounced on single-step tasks. This may indicate that: (a) the current CG architecture doesn't explicitly model temporal dynamics, or (b) the flat MLP benefits from learning the full input-output mapping end-to-end for longer sequences. Future work should explore recurrent/temporal CG variants.

### H1.412: Action-Conditioned Multi-Object Interaction Prediction — Round 181

**Hypothesis**: CG advantage emerges when task requires reasoning about object-object interactions that are action-conditioned (pushing A affects B).

**Method**: Physics simulator with contact-based multi-object dynamics. Task: predict final positions after push actions.

**Results**:
| Objects | Baseline Loss | CG Loss | Improvement | CG Wins |
|---------|--------------|---------|-------------|---------|
| 3 | 0.001579 | 0.000255 | +83.85% | ✓ |
| 5 | 0.003654 | 0.000247 | +93.23% | ✓ |
| 7 | 0.005316 | 0.000286 | +94.62% | ✓ |
| 10 | 0.007891 | 0.000312 | +96.05% | ✓ |

**Key Finding**: **SUPPORTED.** CG advantage scales with object count: +84% (3 obj) → +93% (5 obj) → +95% (7 obj) → +96% (10 obj). Baseline loss grows 4x with complexity while CG stays constant.
