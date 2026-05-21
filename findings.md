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

### H1.470.1.1.44: Larger Hidden Dimensions & Activation Functions — Round 284 (REFUTED)

**Context**: H1.470.1.1.43 REFUTED - all architectural modifications show 100% underfitting. This experiment tests whether larger hidden dimensions (128, 256, 512) and modern activation functions (GELU, SiLU) can reduce underfitting.

**Hypothesis**: Larger hidden dimensions and modern activation functions (GELU, SiLU) will reduce underfitting below 67.5%.

**Configurations Tested**:
- Hidden dimensions: [128, 256, 512]
- Activation functions: [ReLU, GELU, SiLU]
- Number of layers: [2, 4]
- Total configurations: 18

**Key Findings**:

1. **All Configurations Show High Underfitting** (100% underfit):
   | Configuration | Val Loss | Underfit % |
   |--------------|----------|------------|
   | Best (128h, SiLU, 4L) | 1.1607 | 117.6% |
   | Worst (512h, SiLU, 4L) | 1.5888 | 158.9% |

2. **Smaller Hidden Dimensions Perform Better**:
   | Hidden Dim | Avg Val Loss | Avg Underfit % |
   |------------|--------------|----------------|
   | **128** | **1.2752** | **130.5%** |
   | 256 | 1.4383 | 149.4% |
   | 512 | 1.4458 | 147.2% |

3. **Modern Activations Slightly Better Than ReLU**:
   | Activation | Avg Val Loss | Avg Underfit % |
   |------------|--------------|----------------|
   | ReLU | 1.4271 | 145.6% |
   | GELU | 1.3670 | 140.8% |
   | **SiLU** | **1.3652** | **140.7%** |

4. **Conclusion**: Larger hidden dimensions do NOT help - they actually worsen underfitting. The 128-dim with SiLU is best at 117.6% underfit, but still severely underfitting. The problem is NOT about model capacity - it's fundamentally about the data representation or the learning objective.

---

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

### H1.470.1.1.42: Extreme Learning Rates & Optimizers — Round 282 (REFUTED)

**Context**: H1.470.1.1.41 showed some underfitting reduction with aggressive training. This experiment tests extreme learning rates (0.01-0.1) and alternative optimizers.

**Hypothesis**: Extreme learning rates and modern optimizers (AdamW, SGD with momentum) will reduce underfitting.

**Key Findings**:
- Higher learning rates (0.1) cause severe instability
- Best results at conservative LR=0.001
- Underfitting persists across all optimizer choices
- Conclusion: REFUTED - training hyperparameter tuning alone cannot solve the underfitting

---

### H1.470.1.1.41: Aggressive Training Strategies — Round 281

**Key Findings**:
- Tested 72 configurations with various LR, epochs, schedules
- Lower LR (0.0001) with more epochs (200) shows best results
- Some configurations achieve val_loss < 0.1 on simple tasks
- Underfitting still present on complex tasks
- Conclusion: INCONCLUSIVE - training helps but doesn't fully solve the problem
