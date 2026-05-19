# Round 210 Summary — H1.444: Architectural Modifications Fix GraphCG Underperformance

## What We Did

After H1.443 showed GraphCG underperforms MLP across ALL conditions (-7.2% to -33.7%), we tested 4 architectural modifications to fix the deficit: edge-aware attention, high-dimensional object representations (8→32 dim), residual connections, and a combined approach.

## Key Results

**The combined modification achieves +2.6% improvement over MLP** on action prediction tasks — the first time GraphCG has beaten MLP on this task type. High-dimensional object representations (+2.4%) and residual connections (+1.7%) are the key individual fixes. Edge-aware attention alone doesn't help (-1.6%). The combined approach beats MLP at 2-5 objects (+0.5% to +1.5%) but loses at 7 objects (-1.3%).

## Implications

The original GraphCG architecture was fundamentally flawed for action prediction due to representational bottlenecks (8-dim objects) and optimization difficulties. With the right modifications, GraphCG can achieve modest advantages over MLP, but the advantage doesn't scale to high object counts. Next round (H1.445) will test the combined architecture on the full LIBERO task suite to verify transferability.
