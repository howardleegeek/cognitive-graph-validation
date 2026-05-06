# H3.42 Protocol: GWM-Style Action Nodes

## Hypothesis Statement
GWM-style action nodes improve task representation for language-conditioned robotic manipulation.

## Literature Background
The Graph World Model (GWM, ICML 2025) introduces action nodes that represent tasks as graph nodes, linked to state nodes via direct reference or similarity computation. This enables unified modeling across diverse tasks including manipulation.

## Experiment Design

### Architecture
1. **State Graph**: Model world state as graph with object nodes and their relationships
2. **Action Nodes**: Add explicit action/task nodes that link to relevant state nodes
3. **Message Passing**: Aggregate action context through graph neural network

### Implementation
- Test with action conditioning vs without
- Test different link methods (reference vs similarity)
- Compare to baseline concatenation

## Expected Outcome
If GWM-style action nodes help, should see improvement on complex compositional tasks requiring task understanding.

## Status
PENDING - Awaiting execution