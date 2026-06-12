#!/usr/bin/env python3
"""Tests for SPM integration experiment (Round 292)."""

import sys
import json
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from experiment import simulate_training, run_experiment


def test_simulate_training_cg_lower_than_hm():
    """CG should have lower underfit than HM at both lengths."""
    for seq_len in [30, 50]:
        cg = simulate_training("CG", seq_len)
        hm = simulate_training("HM", seq_len)
        assert cg["underfit"] < hm["underfit"], \
            f"CG underfit ({cg['underfit']}) should be < HM ({hm['underfit']}) at seq_len={seq_len}"
        assert cg["loss"] < hm["loss"], \
            f"CG loss ({cg['loss']}) should be < HM ({hm['loss']}) at seq_len={seq_len}"
    print("PASS: test_simulate_training_cg_lower_than_hm")


def test_spm_between_hm_and_cg():
    """SPM should perform between HM and CG (closer to CG due to structural prior)."""
    for seq_len in [30, 50]:
        cg = simulate_training("CG", seq_len)
        hm = simulate_training("HM", seq_len)
        spm = simulate_training("SPM", seq_len)
        assert cg["underfit"] <= spm["underfit"] <= hm["underfit"], \
            f"SPM underfit ({spm['underfit']}) should be between CG ({cg['underfit']}) and HM ({hm['underfit']})"
        assert cg["loss"] <= spm["loss"] <= hm["loss"], \
            f"SPM loss ({spm['loss']}) should be between CG ({cg['loss']}) and HM ({hm['loss']})"
    print("PASS: test_spm_between_hm_and_cg")


def test_reproducibility():
    """Same seed should produce identical results."""
    import random
    random.seed(42)
    r1 = simulate_training("SPM", 30)
    random.seed(42)
    r2 = simulate_training("SPM", 30)
    assert r1 == r2, f"Results not reproducible: {r1} vs {r2}"
    print("PASS: test_reproducibility")


def test_results_file_exists():
    """Experiment should produce a results file."""
    results_path = Path(__file__).parent.parent / "results" / "metrics.json"
    assert results_path.exists(), f"Results file not found: {results_path}"
    with open(results_path) as f:
        data = json.load(f)
    assert "seq_len_30" in data
    assert "seq_len_50" in data
    assert "SPM" in data["seq_len_30"]
    assert "ratio_spm_cg_underfit" in data["seq_len_30"]
    print("PASS: test_results_file_exists")


def test_ratios_computed_correctly():
    """Ratios should be CG-normalized underfit rates."""
    results = run_experiment(30)
    expected_ratio = round(results["SPM"]["underfit"] / results["CG"]["underfit"], 2)
    assert results["ratio_spm_cg_underfit"] == expected_ratio, \
        f"Ratio mismatch: {results['ratio_spm_cg_underfit']} vs {expected_ratio}"
    print("PASS: test_ratios_computed_correctly")


if __name__ == "__main__":
    test_simulate_training_cg_lower_than_hm()
    test_spm_between_hm_and_cg()
    test_reproducibility()
    test_results_file_exists()
    test_ratios_computed_correctly()
    print("\nAll tests passed.")
