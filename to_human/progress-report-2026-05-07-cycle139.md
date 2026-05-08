# Progress Report - May 7, 2026 (Cycle 139)

## Research Status

**Project**: Cognitive Graph: Unified World Model and LLM Architecture
**Domain**: World Models
**Started**: April 7, 2026
**Status**: Active

## Current Experiments

### H1.143: Action-Gated + Decay Attention on Complex Multi-Step Tasks
- **Status**: ❌ REFUTED
- **Result**: -41.1% average improvement
- **Finding**: Action-gated + decay attention underperforms on complex multi-step tasks in synthetic setting

### H1.144: Hybrid Concatenation/Attention Architecture
- **Status**: ❌ REFUTED  
- **Result**: -4.3% average improvement
- **Finding**: Hybrid architecture does not improve over baseline; concatenation consistently outperforms attention

## Research Summary

| Hypothesis | Status | Key Finding |
|------------|--------|-------------|
| H1 (Unified vs Baseline) | ✅ SUPPORTED | +25.6% on real robot data |
| H1.41-50 (Attention) | ✅ SUPPORTED | +99% on real robot tasks |
| H1.142-144 (Attention synthetic) | ❌ REFUTED | -41% to -2064% on synthetic |
| H2.x (Graph structure) | ✅ SUPPORTED | +56-75% on temporal reasoning |
| H3 (Attention) | ❌ simple, ✅ complex | Task-dependent |

**Total**: 25+ SUPPORTED, 2 INCONCLUSIVE, 14 REFUTED

## Key Insights

1. **Attention works on real robot data**: +99% in H1.41, H1.51 (real robot)
2. **Attention fails on synthetic data**: -41% to -2064% in H1.142-144
3. **Graph structure works**: +56-75% on temporal reasoning (H2.x)
4. **Key difference**: Real robot data has inherent temporal structure; synthetic data lacks this structure

## Next Directions

1. Test graph structure on complex multi-step tasks (H2.x showed +56-75%)
2. Test invariant learning for transfer (H1.8 showed +5.4%)
3. Focus on real robot validation rather than synthetic

## Git Commit

- Commit: `4d4d65c` - Add H1.143 and H1.144 experiment results
- Pushed to: https://github.com/howardleegeek/cognitive-graph-validation