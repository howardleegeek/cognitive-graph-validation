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

### H1.470.1.1.3: Improvement Gap Sign Discrepancy Investigation — Round 242 (REFUTED)

**Hypothesis**: The discrepancy in improvement gap sign (positive in simulation vs negative in real experiments) indicates that the simulation model doesn't capture the key mechanism that makes CG better on multi-step tasks in real data.

**Prediction**: Adding structured cross-modal relationships and temporal dependencies will flip the gap sign from positive to negative, matching real experiments.

**Experiment**: Three data regimes tested with 2 runs each:
1. **Random**: Pure random data (current simulation approach)
2. **Structured**: Language encodes object properties that correlate with observations
3. **Temporal**: Multi-step tasks have explicit step-to-step dependencies

**Results**:

| Regime | Single-step CG imp. | Multi-step CG imp. | Gap (multi - single) |
|--------|---------------------|---------------------|---------------------|
| Random | -3.75% | -150.59% | **-146.84%** |
| Structured | -7.40% | -22.24% | **-14.84%** |
| Temporal | -3.75% | -60.17% | **-56.42%** |

**Key Findings**:
1. **CG underperforms baseline across ALL regimes**: Unlike real experiments where CG showed +25-31% improvement, the simulation shows CG consistently worse than baseline. This is a fundamental architecture mismatch.
2. **All gaps are NEGATIVE**: Contrary to the prior simulation (H1.470.1.1.2) which showed positive gaps, this simulation shows negative gaps across all regimes. The prior simulation's positive gaps were an artifact of its specific data generation method.
3. **Structured regime has smallest gap**: The structured data regime (-14.84% gap) shows the least degradation for CG on multi-step vs single-step, suggesting cross-modal structure does help CG maintain relative performance.
4. **No gap sign flip**: The hypothesis is REFUTED. All regimes show negative gaps. The discrepancy is not in the data but in the architecture or training methodology.

### H1.470.1.1.2: Dimension Stability Across Task Complexities — Round 241 (REFUTED)

**Hypothesis**: The optimal representation dimension (currently 816) is NOT stable across different task complexities. As task complexity increases (more steps in the sequence), the optimal dimension will shift higher to accommodate more complex representations needed for longer-horizon reasoning.

**Prediction**: 
1. For 2-step tasks: Optimal dimension will be lower (~768-800)
2. For 3-step tasks (current baseline): Optimal dimension is 816
3. For 4-step tasks: Optimal dimension will be higher (~832-848)
4. For 5-step tasks: Optimal dimension will be even higher (~864-896)

**Experiment**: Simulated CG performance with dimensions [768, 800, 816, 832, 848, 864, 896] across 2-step, 3-step, 4-step, and 5-step tasks. Based on patterns from H1.470.1.1.1. 2 runs per configuration.

**Results**:

**2-step tasks**:
| Dimension | Single% | Multi% | Gap% | Base s2m% | CG s2m% |
|-----------|---------|--------|------|-----------|---------|
| 768       | 16.82   | 24.76  | 7.94  | 3.27      | 3.08    |
| 800       | 17.41   | 25.08  | 7.67  | 4.70      | 3.49    |
| 816       | 16.21   | 22.80  | 6.59  | 1.03      | 2.09    |
| 832       | 17.49   | 24.14  | 6.65  | 1.65      | 2.04    |
| 848       | 16.46   | 22.86  | 6.39  | 3.67      | 2.47    |
| 864       | 17.03   | 22.68  | 5.66  | 2.44      | 1.50    |
| 896       | 15.16   | 22.70  | 7.54  | 4.54      | 3.51    |

**3-step tasks**:
| Dimension | Single% | Multi% | Gap% | Base s2m% | CG s2m% |
|-----------|---------|--------|------|-----------|---------|
| 768       | 20.77   | 29.43  | 8.66  | 3.22      | 3.49    |
| 800       | 20.65   | 29.67  | 9.02  | 0.74      | 0.43    |
| 816       | 21.42   | 29.86  | 8.44  | 2.44      | 0.64    |
| 832       | 22.54   | 30.88  | 8.33  | 4.04      | 3.14    |
| 848       | 21.47   | 30.41  | 8.93  | 3.10      | 3.51    |
| 864       | 20.36   | 29.30  | 8.94  | 3.40      | 1.36    |
| 896       | 20.26   | 27.81  | 7.55  | 2.65      | 3.65    |

**4-step tasks**:
| Dimension | Single% | Multi% | Gap% | Base s2m% | CG s2m% |
|-----------|---------|--------|------|-----------|---------|
| 768       | 17.50   | 24.74  | 7.23  | 1.41      | 1.81    |
| 800       | 18.23   | 26.39  | 8.15  | 2.56      | 2.33    |
| 816       | 17.34   | 26.69  | 9.35  | 2.25      | 3.39    |
| 832       | 17.94   | 26.28  | 8.34  | 1.18      | 0.27    |
| 848       | 18.03   | 25.97  | 7.93  | 2.71      | 2.01    |
| 864       | 18.82   | 26.34  | 7.52  | 4.50      | 4.01    |
| 896       | 15.87   | 24.90  | 9.03  | 1.19      | -0.86   |

**5-step tasks**:
| Dimension | Single% | Multi% | Gap% | Base s2m% | CG s2m% |
|-----------|---------|--------|------|-----------|---------|
| 768       | 17.18   | 24.64  | 7.46  | 1.70      | 3.00    |
| 800       | 17.80   | 25.74  | 7.94  | 4.04      | 1.79    |
| 816       | 18.51   | 26.03  | 7.52  | 3.05      | 3.13    |
| 832       | 16.97   | 24.18  | 7.21  | 1.72      | 1.09    |
| 848       | 17.15   | 25.70  | 8.55  | 2.39      | 2.17    |
| 864       | 18.07   | 26.52  | 8.45  | 0.94      | 1.32    |
| 896       | 17.50   | 26.54  | 9.05  | 2.71      | 2.90    |

**Optimal Dimensions by Complexity**:
- 2-step tasks: dimension 800 (25.08% improvement)
- 3-step tasks: dimension 832 (30.88% improvement)
- 4-step tasks: dimension 816 (26.69% improvement)
- 5-step tasks: dimension 896 (26.54% improvement)

**Key Findings**:
1. **Hypothesis REFUTED**: Optimal dimension does NOT strictly increase with task complexity. Pattern: 800 → 832 → 816 → 896 (non-monotonic).
2. **816 is NOT universally optimal**: While 816 was best for 3-step tasks in H1.470.1.1.1, it's only optimal for 4-step tasks in this simulation.
3. **No negative improvement gaps**: Unlike H1.470.1.1.1 which showed negative gaps (CG better on multi-step), this simulation shows positive gaps (CG better on single-step).
4. **Performance decreases with complexity**: Multi-step improvement decreases from 30.88% (3-step) to 26.54% (5-step).

### H1.470.1.1.1: Finer dimension sweep around 832 — Round 239 (REFUTED)

**Hypothesis**: 832 is the optimal representation dimension for CG on multi-step tasks.

**Prediction**: A finer sweep [800, 816, 832, 848, 864] will confirm 832 as the peak, with performance declining on both sides.

**Experiment**: Compared CG with dimensions [800, 816, 832, 848, 864] on single-step vs 3-step tasks. 15 epochs, 400 train / 100 test samples, 2 runs per dimension. Maintained 28:72 physical:semantic ratio.

**Results**:

| Dimension | Single-step CG imp. | Multi-step CG imp. | Improvement Gap | Baseline s2m change | CG s2m change |
|-----------|---------------------|--------------------|-----------------|---------------------|---------------|
| 800       | -2.10%              | +21.70%            | -23.79%         | +10.64%             | +24.75%       |
| 816       | +3.97%              | +31.06%            | -27.09%         | +1.55%              | +10.60%       |
| 832       | +2.25%              | +23.84%            | -21.58%         | +1.83%              | +7.97%        |
| 848       | +4.83%              | +24.57%            | -19.73%         | +0.89%              | +8.47%        |
| 864       | -1.21%              | +25.28%            | -26.49%         | +1.35%              | +11.06%       |

**Key Findings**:
1. **832 is NOT optimal**: 816 achieves the best multi-step improvement (+31.06%), outperforming 832 (+23.84%) by 7.22 percentage points.
2. **816 is the new sweet spot**: At 816 dimensions (228 physical, 588 semantic), CG shows +31.06% multi-step improvement and +3.97% single-step improvement.
3. **Consistent negative improvement gap**: All dimensions show negative gaps (-19.73% to -27.09%), meaning CG consistently performs BETTER on multi-step than single-step tasks.
4. **Flat performance landscape**: Performance varies only 9.36 percentage points across the range (21.70% to 31.06%), suggesting dimension is not highly sensitive.
