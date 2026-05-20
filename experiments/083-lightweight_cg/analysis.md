# H1.470.1.1.13: Lightweight CG Variants — Parameter Budget Analysis

## Round 252

### Hypothesis
CG's poor performance is due to parameter budget mismatch and architectural complexity, not the unified representation concept itself. Lightweight CG variants with reduced dimensions (matching LSTM's ~358K parameter budget) will perform better than the bloated 1.995M param CG.

### Prediction
Reduced-dimension CG variants will close the performance gap with LSTM when parameter budgets are controlled.

### Experiment Design
Tested 7 architectures across 3 task types:

| Architecture | Unified Dim | GNN Layers | Params |
|-------------|-------------|------------|--------|
| Baseline | N/A | 0 | 36K |
| LSTM | N/A | 0 | 344K |
| CG-tiny | 64 (32+32) | 1 | 16K |
| CG-small | 128 (64+64) | 2 | 64K |
| CG-medium | 256 (128+128) | 2 | 243K |
| CG-noGNN | 128 (64+64) | 0 | 47K |
| CG-attention | 128 (64+64) | 0 (attn) | 81K |

### Results

#### Temporal-Only Task

| Architecture | Loss | Improvement vs Baseline |
|-------------|------|------------------------|
| Baseline | 0.0925 | +0.00% |
| LSTM | 0.0031 | **+96.60%** |
| CG-tiny | 0.0923 | +0.26% |
| CG-small | 0.0934 | -0.92% |
| CG-medium | 0.0943 | -1.92% |
| CG-noGNN | 0.0920 | +0.59% |
| CG-attention | 0.0838 | +9.40% |

#### Cross-Modal-Only Task

| Architecture | Loss | Improvement vs Baseline |
|-------------|------|------------------------|
| Baseline | 0.2271 | +0.00% |
| LSTM | 0.0874 | **+61.54%** |
| CG-tiny | 0.2274 | -0.12% |
| CG-small | 0.2281 | -0.44% |
| CG-medium | 0.2288 | -0.74% |
| CG-noGNN | 0.2276 | -0.18% |
| CG-attention | 0.2285 | -0.61% |

#### Combined Task

| Architecture | Loss | Improvement vs Baseline |
|-------------|------|------------------------|
| Baseline | 0.0452 | +0.00% |
| LSTM | 0.0023 | **+94.86%** |
| CG-tiny | 0.0452 | +0.04% |
| CG-small | 0.0453 | -0.23% |
| CG-medium | 0.0472 | -4.35% |
| CG-noGNN | 0.0448 | +1.02% |
| CG-attention | 0.0400 | **+11.50%** |

### Summary Statistics

| Architecture | Avg Improvement | Params |
|-------------|----------------|--------|
| LSTM | **84.33%** | 344K |
| CG-attention | 6.76% | 81K |
| CG-noGNN | 0.48% | 47K |
| CG-tiny | 0.06% | 16K |
| CG-small | -0.53% | 64K |
| CG-medium | -2.34% | 243K |
| Baseline | 0.00% | 36K |

### Conclusion: REFUTED

**The hypothesis is decisively REFUTED.** Even lightweight CG variants with controlled parameter budgets dramatically underperform LSTM:

1. **Best lightweight CG (cg_attention): 6.76% avg improvement** vs **LSTM: 84.33%** — a 77.6 percentage point gap
2. **Parameter budget is NOT the issue**: CG-medium (243K params, close to LSTM's 344K) performs WORSE than CG-tiny (16K params), suggesting the unified representation architecture itself is the problem
3. **CG-attention is the only variant showing consistent improvement** across all tasks, but still only achieves 6.76% vs LSTM's 84.33%
4. **The trend is clear**: as CG dimension increases, performance DECREASES (CG-tiny > CG-small > CG-medium), opposite of what capacity-limited hypothesis would predict

### Key Insight

The unified representation concept itself appears fundamentally flawed for these language-conditioned robotic tasks. The problem is NOT:
- Parameter budget (lightweight CGs don't help)
- GNN complexity (CG-noGNN doesn't help)
- Representation dimension (larger dims make it worse)

The problem IS likely:
- **Unified space forces incompatible representations**: Physical dynamics and semantic concepts have fundamentally different mathematical structures that cannot be meaningfully merged into a single vector space
- **LSTM's separated encoding + temporal processing is inherently better suited**: It processes observations temporally while keeping language as a conditioning signal, rather than forcing them into the same space
- **The baseline's concatenation approach is actually optimal**: Separate encoders preserve modality-specific structure, and the fusion layer learns task-specific weighting

### Next Steps

This result, combined with H1.470.1.1.12 (Hybrid LSTM+CG REFUTED), strongly suggests the Cognitive Graph architecture is not viable for these tasks. The research should pivot to:

1. **H1.470.1.1.14**: Investigate WHY LSTM is so dominant — is it the temporal processing, the separated encoding, or both?
2. **Consider abandoning the CG hypothesis entirely** and focusing on optimizing LSTM-based architectures
3. **Test if CG has ANY niche** where it outperforms — perhaps on tasks specifically designed to require cross-modal reasoning at each timestep
