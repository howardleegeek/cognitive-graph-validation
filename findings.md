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

### H1.470.1.1: Fine-grained dimension sweep around 768 — Round 238 (REFUTED)

**Hypothesis**: There exists an optimal representation dimension (~768) for CG on multi-step tasks.

**Prediction**: Fine-grained sweep around 768 [640, 704, 768, 832, 896] will show peak multi-step improvement at 768.

**Experiment**: Compared CG with dimensions [640, 704, 768, 832, 896] on single-step vs 3-step tasks. 15 epochs, 800 train / 200 test samples, 3 runs per dimension.

**Results**:

| Dimension | Single-step CG imp. | Multi-step CG imp. | Improvement Gap | Baseline s2m change | CG s2m change |
|-----------|---------------------|--------------------|-----------------|---------------------|---------------|
| 640       | +35.18%             | +28.94%            | +6.23%          | +24.23%             | +35.10%       |
| 704       | +39.29%             | +33.76%            | +5.53%          | +21.55%             | +32.41%       |
| 768       | +29.87%             | +31.06%            | -1.19%          | -21.24%             | -26.50%       |
| 832       | +39.73%             | +41.49%            | -1.76%          | -4.94%              | -9.18%        |
| 896       | +34.48%             | +28.88%            | +5.59%          | +5.05%              | +12.89%       |

**Key Findings**:
1. **768 is NOT optimal**: 768 shows only +31.06% multi-step improvement (3rd best out of 5).
2. **832 is the new sweet spot**: 832 dimensions achieves best multi-step performance (+41.49%) AND high single-step performance (+39.73%).
3. **Negative improvement gap at optimal dimensions**: Both 768 (-1.19%) and 832 (-1.76%) show CG performing BETTER on multi-step than single-step.
4. **High variance at 768**: 768 shows highest standard deviation (8.79-9.36%), suggesting instability.
5. **Clear peak at 832**: Performance degrades at 896 (+28.88% multi-step), confirming non-monotonic relationship.

**Implications**:
- Optimal representation dimension appears to be ~832, not 768
- At optimal dimension, CG actually performs BETTER on multi-step tasks than single-step
- Need even finer sweep around 832 to confirm exact optimal point

### H1.470.1: Representation Bottleneck - Dimension Sweep — Round 237 (REFUTED)

**Hypothesis**: CG's advantage decreases with task complexity because the fixed 512-dim unified representation becomes a bottleneck when encoding both current state and task history. Increasing representation dimension should reduce this gap.

**Prediction**: Larger unified representations (768, 1024) will show smaller single-to-multi performance gap for CG.

**Experiment**: Compared CG with dimensions [256, 512, 768, 1024] on single-step vs 3-step tasks. 15 epochs, 800 train / 200 test samples.

**Results**:

| Dimension | Single-step CG imp. | Multi-step CG imp. | Improvement Gap | CG s2m change | Baseline s2m change |
|-----------|---------------------|--------------------|-----------------|---------------|---------------------|
| 256       | +4.50%              | +0.28%             | -4.22%          | +44.36%       | +46.72%             |
| 512       | +0.83%              | +2.67%             | +1.84%          | +47.20%       | +46.20%             |
| 768       | +4.45%              | +8.45%             | +4.00%          | +49.02%       | +46.79%             |
| 1024      | +18.11%             | +8.52%             | -9.59%          | +42.21%       | +48.27%             |

**Key Findings**:
1. **Non-monotonic relationship**: The improvement gap does NOT consistently decrease with dimension. It peaks at 768 (+4.00%) then collapses at 1024 (-9.59%).
2. **768 is the sweet spot**: At 768 dimensions, CG shows its best multi-step performance (+8.45% improvement over baseline), with the largest positive gap (+4.00%).
3. **1024 overfits single-step**: At 1024 dimensions, CG achieves +18.11% on single-step but only +8.52% on multi-step — the gap widens dramatically (-9.59%).
4. **Baseline stable**: Baseline single-to-multi change remains stable around 46-48% across all dimensions, suggesting the effect is specific to CG architecture.