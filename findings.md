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

### H1.430: Attention-Based Temporal Aggregation (Transformer) vs RNN — Round 196

**Hypothesis**: Transformer-based temporal aggregation will outperform GRU for multi-stage tasks because attention can capture long-range temporal dependencies more effectively than sequential RNN processing.

**Prediction**: Transformer will achieve >5% improvement over GRU on multi-stage tasks with sequences of 15+ timesteps.

**Context**: H1.429 showed GRU provides modest improvement (+2.9% over vanilla CG on multi-stage). LSTM failed badly (-23%). This tests whether the attention mechanism itself (not just temporal modeling) is the key factor.

**Method**: Train 5 architectures on multi-stage temporal sequences (15 timesteps, 3 objects, 500 demos): Baseline MLP, Per-Object CG, Per-Object CG + GRU, Per-Object CG + Transformer (temporal attention only), and Full Transformer CG (unified spatio-temporal attention). Compare validation MSE on predicting final action. 3 runs each.

**Results**:

| Architecture | Mean MSE | Std | Δ vs Baseline |
|--------------|----------|-----|---------------|
| Baseline MLP | 0.033725 | 0.000570 | — |
| Per-Object CG | 0.035387 | 0.000730 | +4.93% |
| Per-Object CG + GRU | 0.035238 | 0.000182 | +4.49% |
| Per-Object CG + Transformer | 0.035418 | 0.000222 | +5.02% |
| Full Transformer CG | 0.035052 | 0.000019 | +3.93% |

**Key Comparisons**:
- Transformer vs GRU: +0.51% (Transformer slightly worse)
- Full Transformer vs GRU: -0.53% (Full Transformer slightly better)
- Transformer vs vanilla CG: +0.09% (essentially equivalent)

**Key Findings**:
1. **Transformer does NOT outperform GRU**: The hypothesis is refuted. Transformer-based temporal aggregation (0.035418) performs marginally worse than GRU (0.035238), with only +0.51% difference — well within noise.
2. **Full Transformer CG is most stable**: Lowest variance (σ=0.000019 vs GRU's σ=0.000182), suggesting unified attention provides more consistent learning, but not better absolute performance.
3. **Baseline MLP still wins**: All CG variants underperform the simple MLP baseline by 3.9-5.0%. This is a persistent pattern across experiments.
4. **Attention mechanism is not the bottleneck**: Replacing GRU with attention for temporal modeling yields no meaningful improvement, suggesting the limitation is not in how temporal dependencies are captured, but in the CG architecture itself.

**Analysis**:
- The prediction of >5% improvement for Transformer over GRU is decisively refuted (actual: -0.53% for Full Transformer).
- The extremely low variance of Full Transformer CG (σ=0.000019) is notable — it suggests the unified attention mechanism provides more deterministic optimization, even if the final performance is similar.
- The persistent underperformance of all CG variants vs baseline MLP suggests the graph structure itself may be introducing unnecessary inductive bias for these synthetic tasks.
- **Implication**: The CG advantage may only manifest on tasks with explicit relational structure (e.g., multi-object manipulation with physical interactions), not on the current synthetic temporal prediction tasks.

**Conclusion**: H1.430 **REFUTED**. Transformer-based temporal aggregation does not outperform GRU. The attention mechanism is not the missing piece for improving CG on multi-stage tasks.

---

### H1.429: Temporal Sequence Modeling — Round 195

**Hypothesis**: Adding LSTM/GRU to Per-Object CG will improve multi-step task performance by capturing temporal dependencies that the static graph misses.

**Context**: H1.425 showed Per-Object CG performs worse on multi-stage tasks. H1.427 showed Per-Object CG transfers best across task types (not overfitting). H1.428 showed hybrid architecture doesn't help. This experiment tests whether temporal modeling is the missing piece.

**Method**: Train Baseline MLP, Per-Object CG, Per-Object CG + LSTM, and Per-Object CG + GRU on two task types (spatial_relations and multi_stage). Compare validation MSE on predicting the final action in a 15-step sequence.

**Results**:

| Architecture | Spatial MSE | Spatial Δ vs Baseline | Multi-Stage MSE | Multi-Stage Δ vs Baseline |
|--------------|-------------|----------------------|-----------------|---------------------------|
| Baseline MLP | 0.047317 | — | 0.007269 | — |
| Per-Object CG | 0.047068 | -0.5% | 0.006919 | -4.8% |
| Per-Object CG + LSTM | 0.101632 | +114.8% | 0.008516 | +17.2% |
| Per-Object CG + GRU | 0.052217 | +10.4% | 0.006721 | -7.5% |

**Key Findings**:
1. **GRU helps multi-stage tasks**: Per-Object CG + GRU achieves -7.5% vs baseline on multi-stage (best result), compared to -4.8% for vanilla Per-Object CG
2. **LSTM hurts performance**: LSTM degrades both spatial (+115%) and multi-stage (+17%) performance significantly
3. **GRU improvement over vanilla CG**: +2.9% improvement on multi-stage tasks, but -10.9% worse on spatial tasks
4. **Task-specific benefit**: Temporal modeling helps multi-stage more than spatial (as predicted), but the magnitude is small

**Analysis**:
- GRU provides modest improvement on multi-stage tasks (2.9% over vanilla CG). LSTM fails badly. Temporal modeling helps multi-stage more than spatial tasks, but effect size is limited. Per-Object CG already captures some temporal structure.

**Conclusion**: H1.429 **PARTIALLY_SUPPORTED**. Temporal modeling helps but effect is smaller than expected.

---

### H1.428: Hybrid Architecture — Round 194

**Hypothesis**: Combining Per-Object CG with 2-Node CG will leverage both fine-grained object reasoning and coarse global reasoning.

**Results**:

| Architecture | Spatial MSE | Multi-Stage MSE | Avg MSE |
|--------------|-------------|-----------------|---------|
| Baseline | 1.111649 | 1.059912 | 1.085781 |
| Per-Object CG | 1.138354 | 1.088253 | 1.113304 |
| 2-Node CG | 1.139350 | 1.125620 | 1.132485 |
| Hybrid | 1.133744 | 1.145911 | 1.139827 |

**Key Findings**:
1. **Baseline outperforms all CG variants** on this synthetic task (no structure needed)
2. Hybrid (1.139827) does NOT beat best individual (Per-Object: 1.113304)
3. Per-Object CG performs best among CG variants
4. The additional complexity of hybrid architecture adds overhead without benefit

**Conclusion**: H1.428 **NOT_SUPPORTED**. The hybrid architecture does not outperform individual architectures.

---

### H1.427: Task Type Transfer Learning — Round 193

**Hypothesis**: Per-Object CG learns task-specific features that don't transfer well, while 2-Node CG learns more generalizable representations.

**Context**: H1.425 showed Per-Object CG performs worse on multi-stage tasks. H1.426 showed explicit relational edges hurt performance. This experiment investigates whether Per-Object CG overfits to task-specific features.

**Method**: Train on one task type, evaluate on another. Measure transfer performance.

**Conclusion**: Per-Object CG transfers best across task types, suggesting it does NOT overfit to task-specific features.
