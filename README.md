# 🤖 Autonomous Cognitive Graph Research System

**Fully automated research pipeline** that generates hypotheses, runs experiments, analyzes results, and pushes to GitHub — 24/7 without human intervention.

---

## 🎯 System Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Autonomous Research Engine                                  │
│  ├─ Generate Hypothesis (based on previous results)         │
│  ├─ Create Experiment Code (auto-generated Python)          │
│  ├─ Run Experiment (train models, collect metrics)          │
│  ├─ Analyze Results (compare baseline vs cognitive graph) │
│  ├─ Update State (research-state.yaml, findings.md)         │
│  ├─ Git Commit & Push (auto-commit every experiment)        │
│  └─ Generate Report (HTML progress dashboard)               │
└─────────────────────────────────────────────────────────────┘
                              │
                    Every 20 minutes (cron job)
```

---

## 📊 Current Status

| Metric | Value |
|--------|-------|
| **Total Experiments** | 5 (H1-H4 + 1 auto-generated) |
| **Primary Result** | **+25.6%** sample efficiency improvement |
| **Dataset** | 550 LIBERO-style robot demonstrations |
| **GitHub** | https://github.com/howardleegeek/oyster-world |

### Key Findings

- ✅ **H1**: Unified Cognitive Graph beats Late Fusion by **25.6%**
- ⚠️ **H2**: Explicit structure ≈ Pure neural (1.7% difference)
- ❌ **H3**: Attention < Concatenation (for simple tasks)
- ⚠️ **H4**: Optimal at 25% physical (close to 28% hypothesis)

---

## 🚀 Quick Start

### Manual Run
```bash
cd /Users/howardli/Downloads/oyster/products/oyster-world/research/cognitive-graph-validation
source .venv/bin/activate
python3 auto_research_engine.py
```

### Monitor Status
```bash
./monitor.sh
```

### View Progress
Open `to_human/progress-*.html` in browser for live dashboard.

---

## 🔄 Automation Setup

The system is scheduled to run **every 20 minutes** via launchd (macOS).

### Check Scheduled Jobs
```bash
list_jobs
```

### Run Immediately
```bash
run autoresearch-loop now
```

### Stop Automation
```bash
delete job autoresearch-loop
```

---

## 📁 Directory Structure

```
research/cognitive-graph-validation/
├── auto_research_engine.py          # 🎯 Main automation script
├── monitor.sh                        # 📊 Status monitor
├── research-state.yaml               # 📋 Central state tracking
├── findings.md                        # 📝 Research narrative
├── research-log.md                   # 📜 Decision timeline
├── src/
│   └── data_loader.py               # 🤖 LIBERO dataset loader
├── experiments/
│   ├── H1-unified-vs-baseline/      # ✅ Primary experiment (25.6%)
│   ├── H2-explicit-graph/           # ⚠️ Structure test
│   ├── H3-attention-vs-concat/      # ❌ Fusion mechanism
│   ├── H4-dimension-allocation/     # ⚠️ Optimal allocation
│   └── 001-*/                       # 🔄 Auto-generated experiments
├── data/
│   └── cache/                       # 💾 550 demos dataset
├── to_human/
│   └── progress-*.html              # 📈 Live dashboards
├── paper/
│   └── RESEARCH_REPORT.md           # 📄 Full research report
└── logs/
    └── autoresearch.log            # 📝 Runtime logs
```

---

## 🧠 How It Works

### 1. Hypothesis Generation
Based on previous results, the engine generates new sub-hypotheses:
- H1 success → Test multi-step tasks, longer sequences, larger scale
- H3 failure → Test attention on complex relational reasoning
- H4 close → Fine-grained sweep around 25% optimal

### 2. Auto-Code Generation
Generates complete PyTorch experiment code:
- Baseline architecture (late fusion)
- Cognitive Graph architecture (unified GNN)
- Training loop with validation
- Metrics collection

### 3. Experiment Execution
- Runs both architectures
- Compares validation MSE
- Calculates improvement percentage
- Determines hypothesis status

### 4. Auto-Documentation
- Updates `research-state.yaml` with trajectory
- Appends to `findings.md` with interpretation
- Commits to git with descriptive message
- Pushes to GitHub automatically

### 5. Progress Reporting
Generates HTML dashboard with:
- Total experiment count
- Average improvement
- Recent results with status colors
- Timestamp of last update

---

## 📈 Results Summary

### Primary Experiment (H1 - Real Robot Data)

| Training Demos | Baseline | Cognitive Graph | Improvement |
|---------------|----------|-----------------|-------------|
| 50 | 0.0175 | 0.0133 | **+24.3%** |
| 100 | 0.0166 | 0.0131 | **+21.2%** |
| 200 | 0.0172 | 0.0125 | **+27.1%** |
| 400 | 0.0179 | 0.0125 | **+30.0%** |

**Average: 25.6% improvement**

### Architecture Comparison

| Component | Baseline | Cognitive Graph |
|-----------|----------|-----------------|
| Observation Encoder | Separate 128-dim | Unified 144-dim (physical) |
| Language Encoder | Separate 128-dim | Unified 368-dim (semantic) |
| Fusion | Late concatenation | Early unified 512-dim |
| Processing | MLP | GNN + Cross-modal attention |
| Action Decoder | MLP | MLP |

---

## 🔬 Active Research Directions

The autonomous engine is currently exploring:

1. **Multi-step manipulation** (pick → place → stack)
2. **Longer trajectories** (20 vs 10 timesteps)
3. **Larger scale** (1000+ demonstrations)
4. **Complex relational reasoning** (where attention should win)
5. **Fine-grained dimension sweep** (20-30% physical range)

---

## 🛠️ Technical Details

### Dataset
- **Source**: LIBERO-style robot manipulation
- **Size**: 550 demonstrations
- **Observations**: Proprioception (8-dim: 7 joints + gripper)
- **Language**: Natural language instructions (32-dim embeddings)
- **Actions**: End-effector poses (7-dim: xyz + rotation + gripper)

### Models
- **Baseline**: Separate encoders + late fusion (V-JEPA 2 style)
- **Cognitive Graph**: Unified 512-dim + GNN + attention

### Training
- Optimizer: Adam (lr=3e-4)
- Loss: MSE
- Epochs: 50-100
- Batch size: 16

---

## 📚 Citation

If you use this research:

```bibtex
@misc{oysterworld2026cognitive,
  title={Cognitive Graph: Unified World Model and LLM Architecture},
  author={Li, Howard},
  year={2026},
  organization={Oyster Labs},
  url={https://github.com/howardleegeek/oyster-world}
}
```

---

## 📝 License

MIT License - See LICENSE file

---

**Last Updated**: April 7, 2026  
**Status**: 🟢 Active (running every 20 minutes)  
**Next Check**: Run `monitor.sh` to see current status
