#!/bin/bash
# Research Monitor - Check status of autonomous research

echo "=========================================="
echo "🧠 Cognitive Graph Research Monitor"
echo "=========================================="
echo ""

cd /Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation

# Check scheduled jobs
echo "📅 Scheduled Jobs:"
list_jobs 2>/dev/null || echo "  No jobs scheduled"
echo ""

# Check experiment count
echo "🔬 Experiments:"
EXP_COUNT=$(ls -1 experiments/ 2>/dev/null | wc -l)
echo "  Total: $EXP_COUNT"
echo ""

# Check latest results
echo "📊 Latest Results:"
if [ -f experiments/H1-unified-vs-baseline/results/real_data_metrics.json ]; then
    echo "  H1 (Real Data): $(cat experiments/H1-unified-vs-baseline/results/real_data_metrics.json | grep average_improvement | cut -d: -f2 | tr -d ', ') improvement"
fi
echo ""

# Check git status
echo "📝 Git Status:"
git log --oneline -3 2>/dev/null || echo "  No git history"
echo ""

# Check logs
echo "📜 Recent Log Activity:"
if [ -f logs/autoresearch.log ]; then
    tail -5 logs/autoresearch.log
else
    echo "  No logs yet"
fi
echo ""

echo "=========================================="
echo "✅ Monitor complete"
echo "=========================================="
