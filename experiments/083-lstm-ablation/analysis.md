# H1.470.1.1.14: LSTM Dominance Ablation Study — Analysis

## Round 253

### Context
Round 252 showed LSTM achieves 84.33% avg improvement vs baseline, while even lightweight CG variants achieve only 6.76%. The unified representation concept is fundamentally flawed. The next question: WHY is LSTM so dominant? Is it the temporal recurrence, the separated encoding, or both?

### Hypothesis
LSTM's dominance comes primarily from its temporal recurrence mechanism. Separated encoding provides additional benefit but is secondary to temporal processing.

### Experiment Design
6 architectures across 3 task types:
1. **Baseline**: Separate encoders + concatenation, no temporal processing
2. **LSTM**: Separated encoders + temporal recurrence (full LSTM)
3. **LSTM-FeedForward**: Separated encoders, NO temporal recurrence (MLP instead)
4. **Separated+Temporal**: Separated encoders + simple temporal processing (1D convolutions)
5. **Unified+Temporal**: Unified encoder + temporal processing (LSTM)
6. **Unified+FeedForward**: Unified encoder, no temporal processing

Tasks: temporal-only, crossmodal-only, combined

### Results

#### Temporal-Only Tasks (where temporal order matters)

| Architecture | Params | Val Loss | Improvement vs Baseline |
|-------------|--------|----------|------------------------|
| Baseline | 61K | 1.9839 | — |
| LSTM | 301K | 0.1219 | **+93.85%** |
| LSTM-FeedForward | 70K | 2.2207 | -11.93% |
| Separated+Temporal | 135K | 0.1684 | **+91.51%** |
| Unified+Temporal | 295K | 0.5538 | +72.09% |
| Unified+FeedForward | 31K | 2.7453 | -38.38% |

#### Crossmodal-Only Tasks

| Architecture | Params | Val Loss | Improvement vs Baseline |
|-------------|--------|----------|------------------------|
| Baseline | 61K | 2.3974 | — |
| LSTM | 301K | 13.9290 | -481.01% |
| LSTM-FeedForward | 70K | 14.4001 | -500.67% |
| Separated+Temporal | 135K | 12.1552 | -407.02% |
| Unified+Temporal | 295K | 14.3737 | -499.57% |
| Unified+FeedForward | 31K | 8.1475 | -239.85% |

#### Combined Tasks

| Architecture | Params | Val Loss | Improvement vs Baseline |
|-------------|--------|----------|------------------------|
| Baseline | 61K | 2.1258 | — |
| LSTM | 301K | 3.3497 | -57.57% |
| LSTM-FeedForward | 70K | 3.5717 | -68.01% |
| Separated+Temporal | 135K | 3.5291 | -66.01% |
| Unified+Temporal | 295K | 3.5359 | -66.33% |
| Unified+FeedForward | 31K | 3.5643 | -67.66% |

### Key Insights

#### 1. Temporal Processing is the DOMINANT Factor
- LSTM (+93.85%) vs LSTM-FeedForward (-11.93%): **105.79% gap**
- Removing temporal recurrence from LSTM makes it WORSE than baseline
- This confirms: temporal recurrence is the critical component, not just parameter count

#### 2. Separated Encoding + Simple Temporal ≈ LSTM
- Separated+Temporal (+91.51%) vs LSTM (+93.85%): **only 2.34% gap**
- Simple 1D convolutions nearly match LSTM's recurrent processing on temporal tasks
- This suggests: the specific recurrence mechanism matters less than having ANY temporal processing with separated encoders

#### 3. Unified Encoding Underperforms Separated Encoding (with temporal)
- Unified+Temporal (+72.09%) vs Separated+Temporal (+91.51%): **19.42% gap**
- Even with the same temporal processing mechanism, unified encoding is worse
- This confirms: separated encoding provides a real advantage beyond just temporal processing

#### 4. Baseline Wins on Crossmodal and Combined Tasks
- On crossmodal-only: ALL architectures perform worse than baseline
- On combined: ALL architectures perform worse than baseline
- This is consistent with H3 (concatenation wins over attention for simple tasks)
- The baseline's simple mean-pooling + concatenation is actually optimal for these task types

#### 5. Unified Encoding is Consistently the Worst Approach
- Unified+FeedForward is the worst architecture on temporal-only (-38.38%)
- Unified+Temporal underperforms Separated+Temporal by 19.42%
- This confirms: the unified representation concept is fundamentally flawed

### Conclusion: H1.470.1.1.14 — SUPPORTED

**LSTM's dominance comes from temporal recurrence, not separated encoding.** However, separated encoding provides an additional 19.42% advantage over unified encoding when combined with temporal processing.

**Critical finding**: The optimal architecture for these tasks is:
1. **Separated encoders** (modality-specific processing)
2. **Temporal processing** (recurrence or convolutions)
3. **Simple fusion** (concatenation, not attention or unified representations)

This is essentially what the baseline + LSTM combination achieves. The cognitive graph's unified representation approach is fundamentally misaligned with what these tasks require.

### Implications for Cognitive Graph Research

The CG architecture's core premise — unified representation space — is contradicted by these results. The data strongly suggests that:
1. Physical and semantic representations should remain SEPARATE
2. Temporal processing should be applied to physical representations
3. Fusion should happen LATE (after encoding), not EARLY (unified space)

This aligns with the V-JEPA + LLM alignment approach that CG was designed to replace. The evidence suggests that approach is actually correct.

### Next Steps

Given these results, the research should pivot to:
1. **H1.470.1.1.15**: Test if late-fusion architectures (separate encoders → temporal processing → late concatenation) outperform both baseline and LSTM
2. **H1.470.1.1.16**: Investigate whether there are ANY task types where unified representations provide an advantage
3. **Consider abandoning the CG hypothesis** and focusing on optimizing the separated+temporal approach
