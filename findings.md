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

### H1.445: Combined GraphCG on Full LIBERO Task Suite — Round 211

**Hypothesis**: The combined GraphCG modifications (edge-aware + high-dim + residual) that achieved +2.6% improvement in H1.444 will generalize across multiple task types and object counts.

**Context**: H1.444 showed combined modifications achieve +2.6% improvement on a single action prediction task. This tests whether that improvement transfers to the full LIBERO-style task suite.

**Method**: Test combined GraphCG vs MLP across 16 configurations:
- Task types: pick, place, push, stack
- Object counts: 2, 3, 5, 7
- Parameters: noise=0.05, 500 samples, 50 epochs, 2 trials

**Results**:

| Configuration | MLP MSE | GraphCG MSE | Improvement |
|---------------|---------|-------------|-------------|
| pick, 2 objects | 0.0175 | 0.0248 | **-41.6%** ✗ |
| pick, 3 objects | 0.0188 | 0.0230 | **-22.2%** ✗ |
| pick, 5 objects | 0.0177 | 0.0240 | **-35.4%** ✗ |
| pick, 7 objects | 0.0179 | 0.0233 | **-29.9%** ✗ |
| place, 2 objects | 0.0170 | 0.0224 | **-32.1%** ✗ |
| place, 3 objects | 0.0182 | 0.0250 | **-37.4%** ✗ |
| place, 5 objects | 0.0179 | 0.0230 | **-28.6%** ✗ |
| place, 7 objects | 0.0180 | 0.0243 | **-35.1%** ✗ |
| push, 2 objects | 0.0186 | 0.0225 | **-20.7%** ✗ |
| push, 3 objects | 0.0179 | 0.0249 | **-39.0%** ✗ |
| push, 5 objects | 0.0177 | 0.0229 | **-29.6%** ✗ |
| push, 7 objects | 0.0175 | 0.0232 | **-32.1%** ✗ |
| stack, 2 objects | 0.0171 | 0.0232 | **-35.8%** ✗ |
| stack, 3 objects | 0.0173 | 0.0237 | **-36.9%** ✗ |
| stack, 5 objects | 0.0173 | 0.0236 | **-36.1%** ✗ |
| stack, 7 objects | 0.0180 | 0.0235 | **-30.2%** ✗ |

**Summary**:
- Overall MLP MSE: 0.0178
- Overall GraphCG MSE: 0.0236
- **Overall Improvement: -32.6%** ✗ (GraphCG loses significantly)
- Win Rate: 0/16 (0%)

**Per Task Type**:
- pick: -32.3%
- place: -33.3%
- push: -30.3%
- stack: -34.8%

**Per Object Count**:
- 2 objects: -32.6%
- 3 objects: -33.9%
- 5 objects: -32.4%
- 7 objects: -31.8%

**Finding**: **REFUTED** - The combined GraphCG modifications from H1.444 do NOT generalize across task types. In fact, they perform **significantly worse** than MLP (-32.6% vs +2.6% in H1.444). This suggests:
1. H1.444's positive result may have been due to specific task characteristics or random seed
2. The modifications introduce additional complexity that hurts generalization
3. The graph architecture may be fundamentally unsuitable for these multi-task scenarios

---

### H1.444: Architectural Modifications to Fix GraphCG Underperformance — Round 210

**Hypothesis**: GraphCG's underperformance on action prediction tasks can be fixed by architectural modifications: (1) edge-aware attention, (2) increased object representation dimension, (3) residual connections.

**Context**: H1.443 showed GraphCG underperforms MLP across ALL conditions (-7.2% to -33.7%) with no crossover point. This experiment tests whether specific architectural changes can close or reverse the gap.

**Method**: Compare 4 modified GraphCG variants against MLP baseline on action prediction task (noise=0.05, 3 objects, 500 samples, 2 trials):
- **GraphCG_Original**: baseline from H1.443 (mean-pooling, 8-dim objects, 2 GNN layers)
- **GraphCG_EdgeAware**: pairwise edge-aware message passing instead of mean-pooling
- **GraphCG_HighDim**: increased object representation (8 → 32 dimensions)
- **GraphCG_Residual**: residual connections with scaled updates (0.1×), 3 GNN layers
- **GraphCG_Combined**: all modifications together

**Results**:

#### Baseline Comparison:

| Model | MSE | Improvement vs MLP |
|-------|-----|-------------------|
| MLP | 0.1009 | — |
| GraphCG_Original | 0.1027 | **-1.8%** ✗ |

#### Modification Comparison:

| Modification | MSE | Improvement vs MLP | Improvement vs Original |
|--------------|-----|-------------------|------------------------|
| Edge-aware | 0.1025 | **-1.6%** ✗ | +0.2% |
| High-dim (32) | 0.0985 | **+2.4%** ✓ | +4.1% |
| Residual | 0.0992 | **+1.7%** ✓ | +3.4% |
| **Combined** | **0.0983** | **+2.6%** ✓ | **+4.3%** |

**Finding**: Two modifications successfully cross the threshold: **high-dimensional object representations** (+2.4%) and **residual connections** (+1.7%). The **combined** approach achieves the best result at +2.6% improvement over MLP. However, this advantage holds only at 2-5 objects and disappears at 7 objects (-1.3%).
