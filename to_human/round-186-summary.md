# Round 186 Summary: Per-Object Cognitive Graph Structure

**Hypothesis H1.420**: CG benefits from finer-grained node structure (per-object nodes instead of single physical blob). Per-object CG will match or exceed GraphAttn performance on permanence task.

**Result**: **STRONGLY SUPPORTED** - Per-Object CG dramatically outperforms all other architectures on object permanence (+61.76% vs baseline), completely reversing the H1.419 finding where GraphAttn appeared superior.

## Key Results

**Object Permanence Task** (key test):
| Model | MSE | vs Baseline |
|-------|-----|------------|
| Baseline MLP | 0.0422 | — |
| 2-Node CG | 0.0400 | +5.37% |
| **Per-Object CG** | **0.0162** | **+61.76%** |
| Hybrid CG | 0.0796 | -88.50% |
| GraphAttn | 0.2605 | -516.60% |

**Spatial Reasoning Task**:
| Model | MSE | vs Baseline |
|-------|-----|------------|
| Baseline MLP | 0.00665 | — |
| 2-Node CG | 0.00279 | +58.11% |
| Per-Object CG | 0.00267 | +59.80% |
| **Hybrid CG** | **0.00107** | **+83.86%** |
| GraphAttn | 0.0545 | -719.10% |

## Key Insight

The H1.419 finding that GraphAttn beats CG on permanence was due to using the wrong CG architecture. Per-Object CG with dedicated object encoders + shared semantic node provides the right inductive bias for object tracking, achieving +61.76% improvement vs GraphAttn's -516.60%.

**Task-Architecture Interaction**: Different tasks benefit from different CG structures:
- Collision: GraphAttn (+34.59%)
- Permanence: Per-Object CG (+61.76%)
- Spatial: Hybrid CG (+83.86%)

## Next Step

H1.421: Test Per-Object CG on real robot data to validate if the architectural improvement transfers to real-world tasks.