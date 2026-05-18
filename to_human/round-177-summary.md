# Round 177 Summary — H1.408: What Data Properties Enable CG Benefits?

**Experiment**: Investigated the critical question raised by H1.407's failure — what data properties must exist for CG to provide benefits? Tested three data types: unstructured synthetic data, relational data with explicit object-entity relationships, and structured multi-object data with graph structure.

**Result**: **SUPPORTED**. CG benefits are highly specific to data with explicit relational structure at the right complexity level. On relational data (obs_dim=27), cg_no_gnn achieves +43.05% improvement over baseline, while cg_with_gnn achieves +31.29%. However, on unstructured data (obs_dim=8) and overly complex structured data (obs_dim=40), CG underperforms by -37.51% and -32.32% respectively.

**Key Insight**: There's a "sweet spot" for CG benefits. The architecture's fixed 512-dimensional unified space (144 physical + 368 semantic) works best when data has clear object-entity relationships but isn't overly complex. This explains the contradictory results across experiments: H1.405/H1.406 used data with relational structure, while H1.407 used unstructured data. For real robot tasks, CG will likely benefit from pick-and-place and stacking tasks with clear object relationships, but may struggle with highly complex multi-object manipulation.
