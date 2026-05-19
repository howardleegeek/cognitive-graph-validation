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

### H1.446: Reproduce H1.444 with More Trials — Round 212

**Hypothesis**: The +2.6% improvement from H1.444 (combined GraphCG modifications) is reproducible and not a statistical anomaly.

**Context**: H1.445 showed -32.6% on the full LIBERO suite, contradicting H1.444's +2.6%. Need to verify if H1.444's result was real.

**Method**: Run exact same config as H1.444 but with 5 trials (vs original 2):
- Combined modifications: edge_aware + high_dim + residual
- Task: action_prediction
- n_trials: 5
- epochs: 50
- batch_size: 64
- noise: 0.05
- n_samples: 500

**Results**:

| Trial | MLP MSE | GraphCG MSE | Improvement |
|-------|---------|-------------|-------------|
| 1 | 0.1110 | 0.0981 | **+11.55%** ✓ |
| 2 | 0.1157 | 0.1073 | **+7.28%** ✓ |
| 3 | 0.1059 | 0.1009 | **+4.75%** ✓ |
| 4 | 0.1075 | 0.0976 | **+9.24%** ✓ |
| 5 | 0.1068 | 0.1030 | **+3.56%** ✓ |

**Summary**:
- Average MLP MSE: 0.1094 ± 0.0036
- Average GraphCG MSE: 0.1014 ± 0.0036
- **Average Improvement: +7.28% ± 2.91%** ✓
- Win Rate: 5/5 (100%)

**Conclusion**: **REPRODUCED** - H1.444's result is reproducible and even stronger with more trials (+7.28% vs +2.6%). The GraphCG modifications work well on single tasks.

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

**Key Insight**: The GraphCG modifications work on single tasks (+7.28%) but fail catastrophically on multi-task generalization (-32.6%). This suggests the attention mechanism overfits to specific task patterns and doesn't transfer across task types.

## Research Trajectory

1. **H1.444**: Single task action prediction → +2.6% (SUPPORTED)
2. **H1.445**: Multi-task LIBERO suite → -32.6% (REFUTED)
3. **H1.446**: Reproduce H1.444 → +7.28% (SUPPORTED - confirmed)

**Next Steps**:
- Investigate why GraphCG fails on multi-task generalization
- Test task-specific fine-tuning vs multi-task training
- Consider simpler attention mechanisms for transfer
