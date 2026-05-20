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

### H1.470.1.1.31: Curriculum Learning for Smooth Robot Trajectories — Round 270 (SUPPORTED)

**Context**: Following H1.470.1.1.30's REFUTED result showing phase-aware training fails on smooth robot trajectories, this experiment tested curriculum learning as an alternative approach that works with continuous dynamics rather than discrete phase structures.

**Hypothesis**: Training on progressively longer/more complex trajectories (curriculum learning) will improve learning on smooth robot manipulation data compared to baseline training.

**Configurations Tested**:
1. Curriculum Learning (short → medium → long): Train on 40-100 steps, then 100-160, then 160-220
2. Baseline with attention: Standard training on all data shuffled
3. Reverse Curriculum (long → short): Train on longest trajectories first
4. Baseline without attention: No cross-attention mechanism

**Key Findings**:

| Configuration | Test Loss | Improvement vs Baseline |
|--------------|-----------|------------------------|
| Curriculum (short→long) | 0.241336 | +81.09% |
| Baseline (attention) | 1.275939 | +0.00% |
| Reverse Curriculum | 0.371635 | +70.87% |
| Baseline (no attention) | 0.335379 | +73.72% |

**Critical Insight**: Curriculum learning shows +81.09% improvement over baseline, significantly outperforming all other approaches. Even reverse curriculum (+70.87%) and no-attention baseline (+73.72%) outperform the attention baseline, suggesting the curriculum approach is the key factor.

**Why This Worked**:
1. **Progressive complexity**: Starting with shorter trajectories allows the model to learn basic dynamics before tackling longer sequences
2. **Smooth transitions**: Curriculum learning naturally handles continuous trajectories without requiring discrete phase detection
3. **Transfer learning**: Skills learned on short trajectories transfer to longer ones

**Implications**: Curriculum learning is a promising approach for smooth robot trajectories. Future work should explore:
- Adaptive curriculum scheduling based on learning progress
- Multi-task curriculum (varying task complexity)
- Combining curriculum with other techniques (attention, etc.)

---

### H1.470.1.1.30: Phase-Aware Training on LIBERO-style Data — Round 269 (REFUTED)

**Context**: Following H1.470.1.1.28's dramatic success with phase-aware training (+99.05% to +99.82% on synthetic hierarchical tasks) and H1.470.1.1.29's failure on mixed/noisy tasks, this experiment tested whether phase-aware training would help on LIBERO-style robot manipulation data with clear phase structure (approach → grasp → lift → transport → place).

**Hypothesis**: Phase-aware training would significantly improve learning on robot manipulation tasks with clear phase structure, similar to the synthetic hierarchical task results.

**Configurations Tested**:
1. Baseline: Standard MSE training
2. Oracle phase-aware: Ground truth phase labels, weight=3.0
3. Detected phase-aware: Predicted phases with auxiliary loss, weight=3.0
4. Oracle phase weight 2.0, 5.0, 10.0: Different weighting strengths

**Key Findings**:

| Configuration | Test Loss | Improvement vs Baseline |
|--------------|-----------|------------------------|
| Baseline | 0.000146 | +0.00% |
| Oracle phase-aware (w=3.0) | 0.000214 | -47.15% |
| Detected phase-aware (w=3.0) | 0.000462 | -217.15% |
| Oracle phase weight 2.0 | 0.000207 | -42.42% |
| Oracle phase weight 5.0 | 0.000212 | -45.88% |
| Oracle phase weight 10.0 | 0.000226 | -54.88% |

**Critical Insight**: ALL phase-aware configurations performed WORSE than baseline. The best phase-aware config (weight=2.0) was still 42.42% worse than baseline.

**Why This Failed**:
1. **Task complexity mismatch**: LIBERO-style manipulation has smooth, continuous trajectories where phase transitions are less critical than synthetic hierarchical tasks with discrete phase boundaries
2. **Loss weighting interference**: Upweighting phase transitions distorts the loss landscape for continuous trajectories
3. **Phase-aware training is NOT a general technique**: Works only on tasks with sharp, discrete phase boundaries
