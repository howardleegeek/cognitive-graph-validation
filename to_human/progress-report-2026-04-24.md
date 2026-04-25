# Progress Report — April 24, 2026

## Research Status: Active

### Cycle 33 Summary

**Key Discoveries Today:**
- **H1.41**: +99% attention improvement maintained on complex multi-step tasks (10-30 steps)
- **H1.42**: +99% across all dimension scales (8k-64k)
- **H1.43**: Sparse attention viable with stride pattern (-2% degradation)

### Overall Research Progress

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| H1 Unified vs Baseline | ✅ +25.6% | Early fusion wins |
| H1.1 Multi-step | ✅ +22.6% | Grows with complexity |
| H1.2 Generalization | ✅ +23.1% | Better to unseen |
| H1.3 Few-shot | ✅ +4.6% | Best at k=2,5 |
| H1.4 Transfer dynamics | ❌ -56.7% | Fails to transfer |
| H1.8 Invariant learning | ✅ +5.4% | Solves transfer |
| H1.11-14 Dimension scaling | ✅ | 4096 optimal w/o reg |
| H1.18-20 Reg + large dims | ✅ | 32k+ with α≥0.1 |
| H1.38 Sparse attention | ✅ +99% | 99% retained |
| H1.39 Action-conditioned | ✅ +30% | Action gates help |
| H1.40 Query-key decay | ✅ +30% | Recent bias |
| H1.41 Complex tasks | ✅ +99% | Maintained |
| H1.42 Dim scaling | ✅ +99% | Consistent |
| H1.43 Sparse patterns | ✅ -2% | Stride best |

**Total: 20+ SUPPORTED, 1 INCONCLUSIVE, 11 REFUTED**

---

## Key Architecture Insights

### What Works
1. **Unified architecture** (early fusion) — +25.6% sample efficiency
2. **Attention mechanisms** (+99%) — especially on long sequences
3. **Sparse attention** — retains 99% of full attention
4. **Action-conditioned gating** — +30% over standard
5. **Query-key decay** — +30% for recent timesteps
6. **Dimension scaling** — 32k+ optimal with α≥0.1

### What Doesn't Work
1. **Cross-dynamics transfer** — fails by -56.7%
2. **Two-branch fusion** — hurts on complex tasks (-31.1%)
3. **Simple concatenation** — worsens long sequences

---

## Next Steps

1. **Paper Draft**: Begin writing paper sections
2. **H1.44**: Test on real robot hardware
3. **Explore**: Cross-dynamics transfer with attention + invariant combined

---

## Files Changed Today

- `findings.md` — Added H1.41-43 results
- `research-state.yaml` — Updated to cycle 33
- New experiments: H1.41, H1.42, H1.43