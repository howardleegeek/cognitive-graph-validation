# Round 173 Summary — H1.404: Coupling × Dim_Ratio Sweep with lr=1e-4

**Experiment**: Re-tested the coupling × dim_ratio parameter sweep (9 configurations) using the optimal learning rate (lr=1e-4) discovered in H1.403, replacing the suboptimal lr=1e-3 used in H1.402.

**Results**: CG won 4/9 configurations (44.4% win rate). Two critical factors emerged: **(1) dim_ratio is the dominant predictor of CG success** — dim_ratio=0.9 won 100% of the time (3/3 configs, avg +12.70% improvement), while dim_ratio=0.1 lost 100% (0/3, avg -15.89%). **(2) Higher coupling between language and observations helps CG** — coupling=0.9 yielded 2/3 wins vs coupling=0.0 at 1/3 wins.

**Interpretation**: The CG architecture requires a sufficiently large unified representation space (high dim_ratio) to realize its advantage over simple concatenation. When the unified space is too small (dim_ratio=0.1), the overhead of cross-modal attention and GNN processing outweighs any representational benefit. This suggests the CG's advantage comes from its ability to model complex interactions in a high-dimensional shared space — not from architectural elegance alone.

**Next step (H1.405)**: Test the optimal configuration (lr=1e-4, dim_ratio=0.9, coupling=0.9) on longer sequences (20+ timesteps) and larger datasets to verify the advantage scales with task complexity.
