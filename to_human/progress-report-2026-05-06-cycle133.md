# Cognitive Graph Research Progress

## Cycle 133 (May 6, 2026)

### Latest Experiment: H3.73 - SSM Gap Test (35-45 timesteps)

**Status: ✅ SUPPORTED (+18.2%)**

| Sequence Length | Baseline MSE | SSM MSE | Improvement |
|-----------------|-------------|--------|-------------|
| 35 | 0.1342 | 0.1169 | +12.9% |
| 40 | 0.1469 | 0.1354 | +7.9% |
| 45 | 0.1651 | 0.1092 | +33.9% |

**Summary**: SSM consistently outperforms baseline on the gap region (35-45 timesteps) where H3.72 showed high variance.

---

## Research Trajectory

### Key Results by Category

| Category | Status | Best |
|----------|--------|------|
| Unified Architecture (H1) | ✅ SUPPORTED | +25.6% on real robot |
| Graph Structure (H2) | ✅ SUPPORTED | +56-75% on temporal |
| SSM/Attention (H3) | ✅ SUPPORTED | +18-99% on complex |
| Transfer Learning | ⚠️ INCONCLUSIVE | H1.8 shows +5.4% |

### Crossover Points Identified

- **20-30 timesteps**: Attention wins (+34%)
- **35-45 timesteps**: SSM wins (+18%)
- **50+ timesteps**: SSM dominates

---

## Hypotheses Status

| ID | Status | Key Finding |
|----|--------|-------------|
| H1 | ✅ +25.6% | Early fusion wins |
| H1.x (attention) | ✅ +18-99% | Scales with complexity |
| H2.x (graph) | ✅ +56-75% | Temporal reasoning |
| H3 | ❌/✅ | Simple: concat, Complex: SSM |
| H3.69-73 | ✅ +18-34% | Medium sequences |

**Total: 25+ SUPPORTED, 1 INCONCLUSIVE, 12 REFUTED**

---

## Next Steps

1. **Paper writing**: Compile findings into paper
2. **Real robot validation**: Test best configs on real robot
3. **New hypotheses**: Explore transfer learning solutions

---

## Research Log

- Cycle 132: H3.72 SSM high variance (+99.8% at 30, -40% at 35)
- Cycle 133: H3.73 SSM gap test (+18.2% on 35-45)
- Commit: 2b552ac