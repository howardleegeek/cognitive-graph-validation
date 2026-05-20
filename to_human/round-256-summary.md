# Round 256 Summary

## Action Taken

**H1.470.1.1.17**: Investigated why Cognitive Graph degrades at 40 timesteps (-10.83% from H1.470.1.1.16) while performing well at 30 timesteps (+85.20%).

## Key Findings

**Root Cause Analysis**: Tested 4 architectures (Baseline, CG Standard, CG+Residual, CG+Strong) across sequence lengths 10-40.

| Architecture | Seq 10 | Seq 20 | Seq 30 | Seq 40 |
|-------------|--------|--------|--------|--------|
| CG Standard | -268%  | -328%  | -365%  | -273%  |
| CG+Residual | -7%    | -19%   | -37%   | -18%   |
| CG+Strong   | +59%   | +53%   | +58%   | +54%   |

**Conclusion**: MIXED - Both error accumulation AND optimization difficulty contribute. The standard CG with high dropout (0.4) severely underfits. The fix: use lower dropout (0.2) + GELU activation → consistent ~55% improvement across all sequence lengths.

## Next Action

**H1.470.1.1.18**: Test CG+Strong architecture on real robot data to validate the optimization fix.
