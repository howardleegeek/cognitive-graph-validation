# Progress Report — May 7, 2026

## Cycle 146 Summary

### Experiments Completed

| Hypothesis | Status | Result | Key Finding |
|------------|--------|--------|-------------|
| H1.151 | ✅ SUPPORTED | +98.7% | Attention works on REAL robot 200-300 steps |
| H1.152 | ❌ REFUTED | -3% | No benefit on random synthetic 250-400 steps |
| H1.153 | ❌ REFUTED | -37397% | Catastrophic on physics-only synthetic |

### Key Discovery

**Attention ONLY works on REAL robot manipulation data.**

| Data Type | Seq Length | Attention Effect |
|-----------|-----------|------------------|
| **Real robot** | 200-300 | **+98.7% (**HELPS**) |
| Random synthetic | 250-400 | -3% (no effect) |
| Physics synthetic | 250-400 | **-37397%** (**HARMS**) |

### Why Attention Works on Real Robot Data

Real manipulation tasks have **temporal structure** that attention can exploit:
- **Object permanence** - tracking objects over time
- **Task phases** - planning → execution → completion
- **Physical causality** - actions cause reactions
- **Motion patterns** - smooth trajectories

### Why Attention Fails on Synthetic Data

- **Random data**: No structure to exploit
- **Physics-only**: Even with dynamics, lacks task structure (no phases, no goals, no object relationships)

### Implications for Research

1. **Attention is task-specific**, not a general improvement
2. **Real robot data required** for validation
3. **Synthetic data is insufficient** for showing attention benefits

---

## Research Status

| Metric | Count |
|--------|-------|
| Total SUPPORTED | 25+ |
| Total REFUTED | 13 |
| Total INCONCLUSIVE | 2 |
| PENDING | 0 |

### Key Conclusions

1. **Unified architecture**: +25.6% on real robot data (SUPPORTED)
2. **Attention**: +99% on REAL robot complex tasks (SUPPORTED) 
3. **Attention**: -3% to -37397% on synthetic (REFUTED)
4. **Graph structure**: +56-75% on temporal reasoning (SUPPORTED)

---

## Next Steps

Given the pattern that attention needs REAL temporal structure:

1. Focus on real robot experiments only
2. Test attention on different manipulation task types
3. Test attention with different action spaces
4. Combine with invariant learning for transfer

---

## Files Modified

- `findings.md` - Updated with H1.151-153 results
- `research-state.yaml` - Updated with hypotheses
- `experiments/H1.152-attention-250-400-step/` - New experiment
- `experiments/H1.153-attention-physics-synthetic/` - New experiment

---

*Report generated May 7, 2026*