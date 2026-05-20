#!/bin/bash
# Run the H1.470.1.1.18 experiment

echo "Starting H1.470.1.1.18: Test CG+Strong architecture on real robot data"
echo "======================================================================"

# Activate virtual environment if exists
if [ -d "../../../venv" ]; then
    source ../../../venv/bin/activate
fi

# Run the experiment
python experiment.py

echo ""
echo "Experiment completed!"
echo "Results saved to ../results/results.json"