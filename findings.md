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

### H1.470.1.1.43: Architectural Modifications — Round 283 (REFUTED)

**Context**: H1.470.1.1.42 REFUTED - extreme LRs worsen underfitting. Key insight: underfitting is architectural, not training-related. This experiment tests whether residual connections, layer normalization, deeper/wider networks can reduce underfitting.

**Hypothesis**: Architectural modifications (residual connections, layer normalization, deeper/wider networks) will reduce underfitting below 67.5%.

**Configurations Tested**:
- Hidden dimensions: [64, 128]
- Number of layers: [2, 4, 6]
- Layer normalization: [True, False]
- Residual connections: [True, False]
- Total configurations: 24

**Key Findings**:

1. **All Configurations Show High Underfitting** (100% underfit):
   | Configuration | Val Loss |
   |--------------|----------|
   | Best (64h, 2L, no LN, no res) | 0.9806 |
   | Worst (128h, 2L, no LN, res) | 1.0794 |

2. **Residual Connections HURT Performance**:
   | Residual | Avg Val Loss |
   |----------|--------------|
   | **False** | **1.0009** |
   | True | 1.0474 |

3. **LayerNorm Has Minimal Impact**:
   | LayerNorm | Avg Val Loss |
   |-----------|--------------|
   | False | 1.0238 |
   | True | 1.0245 |

4. **Depth/Width Trade-off**:
   | Depth | Avg Val Loss |
   |-------|--------------|
   | 2 layers | 1.0306 |
   | **4 layers** | **1.0148** |
   | 6 layers | 1.0271 |

   | Width | Avg Val Loss |
   |-------|--------------|
   | **64 hidden** | **1.0193** |
   | 128 hidden | 1.0290 |

5. **Conclusion**: Architectural modifications (residual, layer norm, depth, width) do NOT solve the underfitting problem. The issue is more fundamental - likely related to the representation capacity of the 512D unified space or the synthetic data generation.

---

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
