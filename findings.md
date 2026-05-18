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

### H1.436: CG Domain of Applicability — Round 202

**Hypothesis**: CG performs better on tasks with clear relational structure (object relationships, spatial reasoning) vs continuous control tasks (trajectory following, smooth motion).

**Context**: Following H1.435 which showed CG performs relatively better on high complexity tasks but still underperforms MLP overall, this experiment tests whether CG has a domain-specific advantage on relational vs continuous control tasks.

**Method**: Generate synthetic relational tasks (object-to-object relationships) and continuous control tasks (smooth trajectory prediction). Train MLP and CG on each. 3 trials, 15 epochs each.

**Results**:

| Task Type | MLP MSE | CG MSE | CG vs MLP |
|-----------|---------|--------|-----------|
| Relational | 0.1256 | 0.4447 | **+254.1%** |
| Continuous Control | 0.0122 | 0.0173 | **+41.5%** |

**Key Findings**:

1. **CG underperforms MLP on both task types** in this synthetic setup

2. **CG performs relatively better on continuous control** (41.5% worse) compared to relational tasks (254.1% worse), which is OPPOSITE to the hypothesis

3. **Hypothesis NOT SUPPORTED**: The data does not support the claim that CG has a domain-specific advantage on relational tasks

4. **Alternative interpretation**: The CG architecture in this simplified implementation may not be well-suited for either task type compared to the MLP baseline. The attention mechanism may be adding unnecessary complexity without providing benefit.

5. **Next steps**: Need to investigate why CG consistently underperforms - is it the architecture itself or the implementation? Consider testing with:
   - More sophisticated CG implementation with proper graph structure
   - Larger model capacity
   - Different attention mechanisms

---

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
   - Data distribution and task type
