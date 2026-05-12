# Research Progress Report - May 11, 2026 (Evening)

## Executive Summary

**H3.100 COMPLETED**: Multi-scale goal decomposition experiment shows **subgoal** (intermediate targets every 5 steps) provides the best performance with **+20.1% average improvement** and **5/5 wins** across all sequence lengths tested.

## Current Research Status

| Hypothesis | Status | Key Finding |
|------------|--------|-------------|
| H1 | ✅ SUPPORTED | +25.6% on real robot data |
| H1.202 | ✅ SUPPORTED | Task structure enables SSM (+37.2%) and Attention (+89.7%) |
| H1.204 | ✅ SUPPORTED | Attention on 50-100 steps (+94.6%) |
| H1.207 | ✅ SUPPORTED | Endpoint goal across complexities (+93.6%) |
| H2 | ⚠️ INCONCLUSIVE | 1.7% difference (noise) |
| H3.91 | ✅ SUPPORTED | Task structure on 20-40 steps (+86.6%) |
| H3.92 | ✅ SUPPORTED | Goal state critical (+87.2%) |
| H3.93 | ✅ SUPPORTED | Full structure on 50-100 steps (+55.6%) |
| H3.94 | ❌ REFUTED | Complex goal representations hurt (-58%) |
| H3.95 | ✅ SUPPORTED | Endpoint on 100+ steps (+95.3%) |
| H3.96 | ✅ SUPPORTED | Endpoint across all ρ levels (+92.8%) |
| H3.97 | ✅ SUPPORTED | Endpoint on 150-250 steps (+31.2%) |
| H3.98 | ✅ SUPPORTED | Hierarchical goal decomposition (+16.4%) |
| H3.99 | ✅ SUPPORTED | Action-consequence modeling (+19.0%) |
| **H3.100** | ✅ **SUPPORTED** | **Subgoal best (+20.1%), multi-scale +6.3%** |
| H4 | 🔸 CLOSE (25%) | 22% physical optimal |

## H3.100 Results: Multi-Scale Goal Decomposition

### Experiment Design
Tested four goal representation strategies across sequence lengths 20-60:
- **Endpoint**: Only final goal state
- **Milestone**: Goals at 25%, 50%, 75%, 100% progress
- **Subgoal**: Goals every 5 steps
- **Multi-scale**: Combination of all above

### Results

| Goal Scale | Avg Δ | Wins |
|------------|-------|------|
| endpoint | -1.2% | 1/5 |
| milestone | +3.2% | 3/5 |
| **subgoal** | **+20.1%** | **5/5** |
| multi_scale | +5.1% | 3/5 |

### Key Findings

1. **Subgoal is the winner**: Intermediate targets every 5 steps provide the best performance with +20.1% average improvement and 5/5 wins across all sequence lengths.

2. **Multi-scale adds value**: Combining endpoint + milestones + subgoals provides +6.3% additional benefit over endpoint alone.

3. **Endpoint alone is insufficient**: Simple endpoint goal shows mixed results (-1.2% average), confirming that more granular goal information helps attention mechanisms.

## H1.208 Results: Ultra-Long Sequence Attention (300-500 steps)

### Experiment Design
Tested three goal representation strategies on ultra-long sequences (300-500 steps):
- **Endpoint**: Only final goal state
- **Subgoal**: Intermediate targets every 20 steps
- **Combined**: Endpoint + multiple subgoals (current + next 2)

### Results

| Goal Type | Avg Δ | Wins |
|-----------|-------|------|
| endpoint | +13.5% | 4/5 |
| subgoal | +30.3% | 5/5 |
| **combined** | **+46.9%** | **5/5** |

### Key Findings

1. **Combined is best**: Endpoint + subgoals provides +46.9% improvement on 300-500 step sequences!

2. **Benefit grows with length**: The advantage of combined goals increases with sequence length.

3. **Major breakthrough**: This is the highest improvement seen on any experiment so far for ultra-long sequences.

## Research Trajectory

### What's Working
- Unified architecture (H1): +25.6% on real robot data
- Task structure enables attention (H1.202, H3.91-93)
- Endpoint goal unlocks attention on long sequences (H3.95-97)
- Hierarchical goal decomposition (H3.98): +16.4%
- Action-consequence modeling (H3.99): +19.0%
- Subgoal decomposition (H3.100): +20.1%
- **Ultra-long attention (H1.208): +46.9%** ← MAJOR BREAKTHROUGH

### What's Not Working
- Simple endpoint goal alone on ultra-long: Only +13.5%
- Complex goal representations (trajectory, keypoint, delta): Hurt performance
- Cross-dynamics transfer: -56.7% (H1.4 REFUTED)

### Next Steps
1. Test H3.101: Combined endpoint + hierarchical + action-consequence
2. Test H1.209: Different attention mechanisms with multi-scale goals
3. Test H1.210: Scaling to 1000+ step sequences

## Git Commit

```
[main 79b8404] research(H3.100): Multi-scale goal decomposition - subgoal best (+20.1%, 5/5 wins)
```

## Summary

Research continues to validate the importance of goal representations for enabling attention mechanisms. The key insight from H3.100 is that **subgoal-level granularity** (intermediate targets every 5 steps) provides the best balance of information and simplicity, outperforming both simple endpoint goals and complex multi-scale approaches.

The research is now exploring the boundaries of how much goal information is optimal, with H3.100 showing that moderate granularity (subgoals) beats both extremes (endpoint alone or full multi-scale).