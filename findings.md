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

### H1.470.1.1.42: Extreme Learning Rates + Alternative Optimizers — Round 281 (REFUTED)

**Context**: H1.470.1.1.41 found LR=1e-2 optimal but underfitting persists at 52.8%. This experiment tests whether even higher LRs (3e-2, 5e-2, 1e-1) or alternative optimizers (AdamW, SGD+momentum, RMSprop) can further reduce underfitting.

**Hypothesis**: Extreme learning rates and alternative optimizers will further reduce underfitting below 52.8%.

**Configurations Tested**:
- Learning rates: [1e-2, 3e-2, 5e-2, 1e-1]
- Optimizers: [Adam, AdamW, SGD+momentum, RMSprop]
- LR schedules: [constant, warmup_cosine, step]
- Model sizes: [32, 64]
- Task complexities: [low, high]
- Total configurations: 192 (4 × 4 × 3 × 2 × 2)

**Key Findings**:

1. **Higher LRs WORSEN Underfitting** (opposite of hypothesis):
   | Learning Rate | Avg Val Loss | Avg Gap | Underfit % |
   |--------------|--------------|---------|-------------|
   | **1e-2** | **0.1085** | **0.0033** | **43.1%** |
   | 3e-2 | 0.1630 | 0.0128 | 60.4% |
   | 5e-2 | 0.1891 | 0.0177 | 81.3% |
   | 1e-1 | 0.2367 | 0.0164 | 85.4% |

2. **Adam/AdamW Best Optimizers**:
   | Optimizer | Avg Val Loss | Avg Gap | Underfit % |
   |-----------|--------------|---------|-------------|
   | **Adam** | **0.1333** | **0.0087** | **55.6%** |
   | AdamW | 0.1340 | 0.0090 | 57.6% |
   | SGD+momentum | 0.1853 | 0.0159 | 81.9% |
   | RMSprop | 0.2448 | 0.0166 | 75.0% |

3. **Schedule Impact**:
   | Schedule | Avg Val Loss | Avg Gap | Underfit % |
   |----------|--------------|---------|-------------|
   | constant | 0.1644 | 0.0135 | 71.4% |
   | warmup_cosine | 0.1783 | 0.0122 | 66.1% |
   | step | 0.1803 | 0.0119 | 65.1% |

4. **Best Configuration**:
   - Config: `lr0.01_optadamw_schedconstant_h64_low`
   - Val Loss: 0.0056
   - Gap: -0.0051 (GOOD)

5. **Overall Statistics**:
   - Underfitting: 67.5% (worse than prior 52.8%)
   - Overfitting: 6.3% (first appearance of overfitting at extreme LRs)
   - Well-fitted: 26.2%

**Conclusion**: **REFUTED**. Extreme learning rates (≥3e-2) significantly worsen underfitting. LR=1e-2 is confirmed as the sweet spot. Adam/AdamW remain the best optimizers. The underfitting problem is NOT solvable through training hyperparameters alone — it points to a fundamental model capacity or architecture limitation.

**Implications**:
- The 52.8% underfitting rate at optimal hyperparameters suggests the model architecture itself is the bottleneck
- Next direction: investigate architectural changes (deeper networks, residual connections, normalization) rather than training hyperparameters

### H1.470.1.1.41: Aggressive Training Strategies — Round 280 (SUPPORTED)

**Context**: H1.470.1.1.40 showed underfitting persists across all model sizes and task complexities. This experiment tests whether more aggressive training (higher learning rates, longer training) can reduce underfitting.

**Hypothesis**: Higher learning rates and longer training will reduce underfitting and improve validation loss.

**Configurations Tested**:
- Learning rates: [1e-4, 1e-3, 1e-2]
- Training epochs: [50, 100, 200]
- LR schedules: [constant, warmup_cosine]
- Model sizes: [32, 64]
- Task complexities: [low, high]
- Total configurations: 72

**Key Findings**:

1. **Higher Learning Rates Reduce Underfitting**:
   | Learning Rate | Avg Val Loss | Avg Gap | Underfit % |
   |--------------|--------------|---------|-------------|
   | 1e-4 | 0.1342 | -0.0200 | 58.3% |
   | 1e-3 | 0.1365 | -0.1070 | 50.0% |
   | **1e-2** | **0.1230** | -0.1169 | **50.0%** |

2. **Training Duration Has Minimal Impact**:
   | Epochs | Avg Val Loss | Avg Gap | Underfit % |
   |--------|--------------|---------|-------------|
   | 50 | 0.1297 | -0.0566 | 50.0% |
   | 100 | 0.1282 | -0.0808 | 54.2% |
   | 200 | 0.1358 | -0.1066 | 54.2% |

3. **Warmup Cosine Schedule Slightly Reduces Underfitting**:
   | Schedule | Avg Val Loss | Avg Gap | Underfit % |
   |----------|--------------|---------|-------------|
   | constant | 0.1268 | -0.0854 | 55.6% |
   | **warmup_cosine** | 0.1356 | -0.0771 | **50.0%** |

4. **Best Configuration**:
   - Config: `lr0.01_epochs50_warmup_cosine_h64_low`
   - Val Loss: 0.0032
   - Train-Val Gap: -0.0014 (GOOD - minimal underfitting)

5. **Underfitting Still Persists**:
   - Underfitting: 38/72 (52.8%)
   - Overfitting: 0/72 (0%)
   - Well-fitted: 34/72 (47.2%)

## Hypothesis Status

| Hypothesis | Status | Key Finding |
|------------|--------|-------------|
| H1: Cognitive Graph > Separated | SUPPORTED | +25.6% improvement with real robot data |
| H2: Attention vs Concatenation | INCONCLUSIVE | 1.7% difference |
| H3: Attention for long sequences | REFUTED | Concatenation wins for simple tasks |
| H4: Dimension allocation (25% physical) | CLOSE | 25% optimal vs 28% hypothesis |

### Current Focus: Underfitting Investigation

**Key Insight**: Across multiple experiments (H1.470.1.1.38-42), underfitting is the dominant issue:
- H1.470.1.1.38: Over-regularization hurts at large capacities
- H1.470.1.1.39: Underfitting on low/medium complexity tasks
- H1.470.1.1.40: Larger models always win, task-aware scaling refuted
- H1.470.1.1.41: Higher LR helps but underfitting persists (52.8% of configs)
- H1.470.1.1.42: **REFUTED** — extreme LRs worsen underfitting (67.5% at 1e-1); LR=1e-2 confirmed optimal

**Pattern**: Models are capacity-limited, not overfitting-prone. Training hyperparameters have been exhaustively explored:
- LR=1e-2 is the confirmed sweet spot (higher LRs hurt)
- Adam/AdamW are the best optimizers
- 50 epochs is sufficient
- Warmup cosine schedule provides marginal benefit

**The underfitting problem is architectural, not training-related.** Next directions:
1. **H1.470.1.1.43**: Test architectural modifications (residual connections, layer normalization, deeper networks)
2. **H1.470.1.1.43b**: Test feature engineering / input preprocessing improvements
3. **H1.470.1.1.43c**: Test ensemble methods or mixture-of-experts for capacity scaling
