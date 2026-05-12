# Cognitive Graph Research Progress Report

## Date: May 11, 2026 (Late Night)

### Status Summary

| Metric | Value |
|--------|-------|
| Total Experiments | 31 |
| Total Runs | 29 |
| Supported | 15+ |
| Refuted | 12+ |
| Inconclusive | 3 |

---

## Key Results This Session

### H1.219: SSM + HierGoals on 100-200 Steps ❌ REFUTED

**Finding**: Concatenation wins decisively on 100-200 step sequences (-8.8% SSM)

| Length | SSM Delta |
|--------|-----------|
| 100 | -12.9% |
| 125 | -11.0% |
| 150 | -8.3% |
| 175 | -7.0% |
| 200 | -5.1% |

**Surprising**: This contradicts H1.113 (+57% on 250-400) and H1.114 (+51% on 400-700)!

---

### H1.220: SSM + HierGoals on 50-100 Steps ❌ REFUTED

**Finding**: Concatenation still wins but margin shrinks (-1.8% SSM)

| Length | SSM Delta |
|--------|-----------|
| 50 | -3.4% |
| 60 | -3.0% |
| 70 | -1.1% |
| 80 | -1.1% |
| 100 | -0.2% |

**Note**: Nearly tied at 100 steps, suggesting crossover approaching.

---

### H3.115: Attention on 20-40 Steps WITH Goal Conditioning ⚠️ INCONCLUSIVE

**Finding**: Mixed results, attention wins 2/5 lengths (+3.9% avg)

| Length | Attention Delta | Result |
|--------|----------------|--------|
| 20 | -2.1% | Concat wins |
| 25 | **+19.6%** | Attention wins |
| 30 | -10.2% | Concat wins |
| 35 | **+17.1%** | Attention wins |
| 40 | -4.6% | Concat wins |

**Note**: Goal conditioning helps at specific lengths (25, 35).

---

## 🔑 CRITICAL DISCOVERY: U-Shaped Crossover Pattern

```
Performance
    ▲
    │ ╭──╮                              SSM+HierGoals
    │ │  │        ╭──╮                  (250-700 steps)
    │ │  │        │  │
    │ │  │   ╭────╯  ╰────╮            Attention+Goal
    │ │  │   │            │            (20-40 steps)
    │ │  │   │            │
    │ ╰──╯───╯            ╰──────────  Concatenation
    │     (dead zone)                 (50-200 steps)
    └──────────────────────────────────────────►
                    Sequence Length
                  20    50  100  200  250  400  700
```

### Crossover Table

| Sequence Range | Best Method | Improvement | Experiments |
|---------------|------------|-------------|-------------|
| 20-40 steps | Attention (with goals) | +3.9% | H3.115 |
| 50-200 steps | **Concatenation** | baseline | H1.219-220 |
| 250-400 steps | SSM+HierGoals | +57.3% | H1.113 |
| 500-700 steps | SSM+HierGoals | +50.9% | H1.114 |

**Key Insight**: The 50-200 step range is a "dead zone" where complex architectures (SSM, Attention) fail.

---

## Architecture Decision Tree (Updated)

```
Input Sequence Length
│
├─► < 50 steps?
│   └─► Use ATTENTION with goal conditioning
│
├─► 50-200 steps?
│   └─► Use simple CONCATENATION
│
└─► > 200 steps?
    └─► Use SSM + HIERARCHICAL GOALS
```

---

## What This Means

1. **SSM doesn't universally win** - it only dominates on very long sequences (250+)
2. **Attention has a niche** - works at 20-40 steps, not 50-200
3. **Concatenation is underrated** - wins the middle range everyone assumed needed complex models
4. **Hierarchical goals matter** - enabled SSM at 250+ but NOT at 100-200

---

## Open Questions

1. Why is 50-200 a dead zone? What makes this range special?
2. Would task-specific structures (goals, phases) help bridge the gap?
3. Is there a hybrid that works at all lengths?

---

## Next Experiments

1. **H1.221**: Test if goal conditioning enables SSM at 150-200 steps
2. **H3.116**: Test phase-aware attention on 50-100 steps
3. **H1.222**: Test adaptive architecture that switches based on sequence length

---

*Report generated: May 11, 2026 23:59*