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

### H1.443: Synthetic vs LIBERO Task Discrepancy Bridge Analysis — Round 209

**Hypothesis**: GraphCG's failure on LIBERO tasks (vs success on synthetic) is due to input representation complexity, task type differences, or data scale relative to model capacity.

**Context**: H1.442 showed GraphCG performs 39.8-44.4% WORSE than MLP on LIBERO tasks, contradicting H1.441's +29.1% improvement on synthetic tasks. This experiment creates a controlled bridge to identify where the advantage disappears.

**Method**: Systematic sweep across 4 dimensions:
1. **Noise level**: 0.0 → 0.2 (clean synthetic → LIBERO-like noise)
2. **Task type**: transformation prediction vs action prediction
3. **Data scale**: 200 → 2000 samples
4. **Object count**: 2 → 7 objects
5. **Combined stress test**: 5 conditions from clean synthetic to LIBERO-hard

**Results**:

#### Noise Sweep (transformation task, 3 objects, 500 samples):

| Noise | MLP MSE | GraphCG MSE | Improvement |
|-------|---------|-------------|-------------|
| 0.00 | 0.0955 | 0.1081 | **-13.1%** ✗ |
| 0.05 | 0.1001 | 0.1171 | **-16.9%** ✗ |
| 0.10 | 0.1033 | 0.1201 | **-16.4%** ✗ |
| 0.15 | 0.1077 | 0.1287 | **-19.5%** ✗ |
| 0.20 | 0.1133 | 0.1339 | **-18.1%** ✗ |

**Finding**: Noise does NOT explain the discrepancy. GraphCG is consistently worse across all noise levels, with no crossover point.

#### Task Type Comparison (noise=0.05, 3 objects, 500 samples):

| Task Type | MLP MSE | GraphCG MSE | Improvement |
|-----------|---------|-------------|-------------|
| transformation | 0.1001 | 0.1171 | **-16.9%** ✗ |
| action | 0.1021 | 0.1365 | **-33.7%** ✗ |

**Finding**: Action prediction is significantly harder for GraphCG (-33.7% vs -16.9%). This suggests the graph architecture struggles with policy-like mappings.

#### Data Scale Sweep (transformation, noise=0.05, 3 objects):

| Samples | MLP MSE | GraphCG MSE | Improvement |
|---------|---------|-------------|-------------|
| 200 | 0.0850 | 0.0919 | **-8.1%** ✗ |
| 500 | 0.1001 | 0.1171 | **-16.9%** ✗ |
| 1000 | 0.0859 | 0.1017 | **-18.4%** ✗ |
| 2000 | 0.0913 | 0.1039 | **-13.8%** ✗ |

**Finding**: More data does NOT help GraphCG catch up. The gap persists or widens with more data.

#### Object Count Sweep (transformation, noise=0.05, 500 samples):

| Objects | MLP MSE | GraphCG MSE | Improvement |
|---------|---------|-------------|-------------|
| 2 | 0.0893 | 0.0998 | **-11.8%** ✗ |
| 3 | 0.1001 | 0.1171 | **-16.9%** ✗ |
| 5 | 0.0912 | 0.1013 | **-11.1%** ✗ |
| 7 | 0.0964 | 0.1033 | **-7.2%** ✗ |

**Finding**: GraphCG's relative performance slightly improves with more objects (-7.2% at 7 objects vs -16.9% at 3), but it never crosses to positive. This hints at a potential scaling benefit that's insufficient to overcome the baseline deficit.

#### Combined Stress Test:

| Condition | Config | MLP MSE | GraphCG MSE | Improvement |
|-----------|--------|---------|-------------|-------------|
| clean_synthetic | noise=0, transform, 500s, 3obj | 0.0955 | 0.1081 | **-13.1%** ✗ |
| noisy_synthetic | noise=0.05, transform, 500s, 3obj | 0.1001 | 0.1171 | **-16.9%** ✗ |
| action_pred | noise=0.05, action, 500s, 3obj | 0.1021 | 0.1365 | **-33.7%** ✗ |
| libero_like | noise=0.1, action, 500s, 5obj | 0.1115 | 0.1453 | **-30.3%** ✗ |
| libero_hard | noise=0.15, action, 300s, 7obj | 0.0971 | 0.1255 | **-29.3%** ✗ |

**Finding**: GraphCG is consistently worse across ALL conditions. The worst performance is on action prediction tasks (-29.3% to -33.7%).

### H1.442: Adaptive Node GraphCG on LIBERO Tasks — Round 208

**Hypothesis**: GraphCG with adaptive node count (n_objects + 2, max 10) will show consistent improvement over MLP baseline on LIBERO-style manipulation tasks, transferring the +29.1% improvement seen in H1.441 synthetic tasks.

**Context**: H1.441 showed that adaptive node count fixes the scaling issue seen in H1.440, with +29.1% average improvement and positive trend (+3.1%/level). This experiment tests whether that finding transfers to LIBERO-style manipulation tasks.

**Method**: Compare MLP-64 vs GraphCG-64-3p with two node configurations:
- Fixed 6 nodes (H1.438 baseline)
- Adaptive nodes (n_objects + 2, max 10)
- 4 task types: simple_pick (2 obj), pick_place (3 obj), multi_object (5 obj), long_horizon (7 obj)
- 400 samples per task, 70/30 train/test split
- 50 epochs, batch size 64, lr 3e-4
- 3 trials per configuration

**Results**:

| Task | Objects | MLP MSE | GraphCG Fixed MSE | GraphCG Adaptive MSE | Fixed vs MLP | Adaptive vs MLP |
|------|---------|---------|-------------------|----------------------|---------------|------------------|
| simple_pick | 2 | 0.0531 | 0.0808 | 0.0749 | **-52.2%** ✗ | **-41.2%** ✗ |
| pick_place | 3 | 0.0855 | 0.1260 | 0.1260 | **-47.4%** ✗ | **-47.3%** ✗ |
| multi_object | 5 | 0.1599 | 0.2266 | 0.2297 | **-41.7%** ✗ | **-43.6%** ✗ |
| long_horizon | 7 | 0.2812 | 0.3734 | 0.4048 | **-32.8%** ✗ | **-43.9%** ✗ |

**Overall Results**:
- MLP baseline MSE: **0.1449**
- GraphCG (fixed 6 nodes) MSE: **0.2027** (39.8% worse than MLP)
- GraphCG (adaptive nodes) MSE: **0.2092** (44.4% worse than MLP)
- Adaptive vs Fixed: **-3.2%** (adaptive slightly worse)

**Key Findings**:

1. **CRITICAL FINDING**: GraphCG performs WORSE than MLP on LIBERO-style manipulation tasks across all task types and node configurations. This contradicts H1.441's +29.1% improvement on synthetic tasks.

2. **No transfer**: The synthetic task advantage does NOT transfer to LIBERO tasks, suggesting fundamental differences in task structure or data characteristics.

3. **Adaptive nodes don't help**: Adaptive node count (H1.441's key innovation) provides no benefit on LIBERO tasks and is slightly worse than fixed nodes.

## Synthesis Across Rounds

### H1 Status: REFUTED for LIBERO-style tasks

The core hypothesis that GraphCG achieves higher sample efficiency than MLP on language-conditioned robotic tasks is **REFUTED** for LIBERO-style manipulation tasks. While H1.441 showed +29.1% improvement on synthetic transformation tasks, H1.442 and H1.443 demonstrate consistent underperformance (-7.2% to -44.4%) on LIBERO-style tasks.

### Key Insights from H1.443 Bridge Analysis:

1. **No single factor explains the discrepancy**: Noise, task type, data scale, and object count individually don't create a crossover point where GraphCG becomes better.

2. **Action prediction is the hardest case**: GraphCG performs worst (-29.3% to -33.7%) on action prediction tasks, which are most similar to real robot control.

3. **Object count shows a hint of scaling**: GraphCG's relative deficit decreases from -16.9% (3 objects) to -7.2% (7 objects), suggesting the graph structure may have benefits at higher complexity that are currently overwhelmed by other factors.

4. **The synthetic task advantage is real but narrow**: GraphCG's success on synthetic tasks appears to be specific to the transformation prediction task structure, not a general advantage.

### Possible Explanations (for future investigation):

1. **Inductive bias mismatch**: GraphCG's object-centric inductive bias may not match the actual structure of LIBERO tasks, which may have more continuous/implicit object relationships.

2. **Optimization difficulty**: The graph architecture may have a harder optimization landscape, requiring more careful initialization, learning rate scheduling, or architectural modifications.

3. **Representation bottleneck**: The fixed object dimension (8) may be insufficient to capture the rich object representations needed for manipulation tasks.

4. **Message passing limitations**: Simple mean-pooling message passing may be insufficient for complex multi-object interactions in manipulation tasks.

## Hypothesis Status Summary

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| H1: GraphCG > MLP on robotic tasks | **REFUTED** (for LIBERO) | H1.442: -39.8% to -44.4%; H1.443: -7.2% to -33.7% |
| H1.441: Adaptive nodes help on synthetic | SUPPORTED | +29.1% improvement on synthetic transformation tasks |
| H1.442: Adaptive nodes help on LIBERO | REFUTED | -44.4% vs MLP |
| H1.443: Bridge analysis | REFUTED | No condition where GraphCG outperforms MLP |
| H2: Attention vs concatenation | Inconclusive | 1.7% difference |
| H3: Attention on long sequences | REFUTED | Concatenation wins for simple tasks |
| H4: Optimal graph size | CLOSE | 25% optimal vs 28% hypothesis |
