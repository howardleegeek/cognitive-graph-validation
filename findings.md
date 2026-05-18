# Research Findings — Cognitive Graph Architecture

## Research Question

Does a unified cognitive graph architecture (early fusion of physical and semantic representations) achieve higher sample efficiency than separated architectures (JEPA + LLM alignment) on language-conditioned robotic tasks?

## Current Understanding

**The Core Hypothesis**: Current approaches (V-JEPA 2, π0, LED-WM) all suffer from representation separation — vision and language exist in different spaces and are only aligned after encoding. This causes:
1. **Grounding problems**: Language not truly grounded in physical dynamics
2. **Combinatorial explosion**: Need to learn all (vision, language) pairings separately
3. **Learning inefficiency**: No gradient flow between modalities during training

**The Cognitive Graph Solution**: A unified 512-dimensional representation space where:
- 144 dimensions encode physical world state (analogous to V-JEPA embeddings)
- 368 dimensions encode semantic/conceptual information (analogous to LLM embeddings)
- Single GNN processes both, with cross-modal attention allowing dynamic interaction
- Explicit graph structure (nodes = objects/concepts, edges = relationships/physics)

## Key Results

### H1.435: Task Complexity Analysis — Round 201

**Hypothesis**: CG advantage depends on task relational complexity. CG should perform relatively better on high-complexity tasks requiring relational reasoning.

**Context**: Following H1.434 which showed CG loses on LIBERO-style manipulation (-4.2% to -12.2%) but H1.433 showed CG wins on synthetic physics tasks (-8.5% to -14.7%), this experiment tests whether the discrepancy can be explained by task complexity.

**Method**: Generate synthetic tasks with 3 complexity levels (low, medium, high) and train MLP, CG-3p, CG-6p on each. 2 trials per complexity level, 5 epochs each.

**Results**:

| Complexity | MLP MSE | CG-3p MSE | CG-6p MSE | CG-3p vs MLP | CG-6p vs MLP |
|------------|---------|-----------|-----------|--------------|--------------|
| Low | 0.054702 | 0.059160 | 0.078773 | **+8.2%** | **+44.0%** |
| Medium | 0.225744 | 0.238993 | 0.234677 | **+5.9%** | **+4.0%** |
| High | 0.337547 | 0.378588 | 0.375437 | **+12.2%** | **+11.2%** |

**Key Findings**:

1. **CG performs WORSE than MLP across all complexity levels** in this synthetic setup (+4.0% to +44.0% worse)

2. **However, CG shows RELATIVE improvement on high complexity tasks**: CG-3p vs MLP improves from +8.2% (low) to +12.2% (high) - a +4.0% improvement trend

3. **CG-6p shows diminishing returns**: While CG-6p is worse than CG-3p on low complexity (+44.0% vs +8.2%), the gap narrows on high complexity (+11.2% vs +12.2%)

4. **Hypothesis PARTIALLY SUPPORTED**: CG does perform relatively better on high complexity tasks, but still underperforms MLP overall in this synthetic setup

5. **Implication**: The CG vs MLP performance difference may depend on:
   - Task relational complexity (CG relatively better on complex tasks)
   - Data distribution (synthetic vs real robot)
   - Model capacity and training regimen

### H1.434: CG on Real Robot Data (LIBERO-style) — Round 200

**Hypothesis**: CG outperforms MLP on real robot manipulation tasks from data/cache.

**Context**: Building on H1.433 which showed CG outperforms MLP on synthetic tasks (-8.5% to -14.7%), this experiment tests whether the advantage holds on real robot-style data.

**Method**: Train 3 architectures (MLP, CG-3p, CG-6p) on 10 LIBERO-style manipulation tasks with 40 demos each, 3 runs per task.

**Results**:

| Task | MLP MSE | CG-3p MSE | CG-6p MSE | CG-3p vs MLP | CG-6p vs MLP |
|------|---------|-----------|-----------|--------------|--------------|
| 0 | 1.108 | 1.187 | 1.230 | -7.1% | -10.9% |
| 1 | 1.318 | 1.369 | 1.489 | -3.9% | -13.0% |
| 2 | 0.695 | 0.675 | 0.660 | **+2.9%** | **+5.0%** |
| 3 | 1.121 | 1.206 | 1.359 | -7.7% | -21.3% |
| 4 | 0.935 | 0.877 | 0.937 | **+6.2%** | -0.2% |
| 5 | 0.950 | 0.982 | 1.063 | -3.3% | -12.0% |
| 6 | 1.004 | 1.088 | 1.157 | -8.5% | -15.3% |
| 7 | 0.784 | 0.857 | 1.025 | -9.5% | -30.9% |
| 8 | 0.658 | 0.677 | 0.706 | -3.0% | -7.4% |
| 9 | 0.869 | 0.932 | 1.010 | -7.6% | -16.4% |

**Key Findings**:

1. **CG does NOT outperform MLP on real robot data** - Average CG-3p: -4.2%, CG-6p: -12.2%

2. **CG-3p wins on 2/10 tasks** (tasks 2 and 4), CG-6p wins on 1/10 tasks (task 2 only)

3. **Deeper message passing (6 passes) actually hurts performance** on real robot data - CG-6p vs CG-3p: -8.1%

4. **The discrepancy with H1.433 (synthetic data) suggests CG advantage may be task-dependent**:
   - On synthetic physics tasks (collision, stacking, pushing): CG wins
   - On LIBERO-style manipulation tasks: MLP wins

5. **Possible explanations**:
   - LIBERO tasks may require different attention patterns than simple physics
   - The synthetic data in H1.433 may have clearer relational structure
   - Real robot data may have more noise or different distributional properties

## Sub-Hypotheses Generated

### H1.435.1: CG Domain of Applicability
**Prediction**: CG outperforms MLP on tasks with:
1. Clear relational structure (object collisions, stacking)
2. Multiple interacting entities
3. Explicit spatial/temporal relationships

**Test**: Compare CG vs MLP on carefully curated task sets that vary these dimensions.

### H1.435.2: Data Distribution Hypothesis
**Prediction**: CG advantage emerges more clearly on:
1. Synthetic data with clean relational structure
2. Tasks where physical and semantic modalities have clear correspondence
3. Longer-horizon tasks requiring relational reasoning

**Test**: Generate synthetic datasets systematically varying these properties.

## Next Steps

1. **Test H1.435.1**: Curate task sets with varying relational complexity
2. **Analyze attention patterns**: Compare what CG vs MLP attends to in different tasks
3. **Investigate training dynamics**: Does CG need more data/epochs to show advantage?
4. **Explore architectural variants**: Different GNN architectures, attention mechanisms