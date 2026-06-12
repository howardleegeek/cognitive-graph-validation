#!/usr/bin/env python3
"""Tests for H1.2.1 LC-SPM experiment."""

import json
import sys
from pathlib import Path

# Import the experiment module
sys.path.insert(0, str(Path(__file__).parent))
from run_lc_spm import (
    simulate_spm_underfit,
    length_conditioned_alpha,
    run_experiment,
)


def test_length_conditioned_alpha_baseline():
    """At ref_len, alpha should equal alpha_base."""
    alpha = length_conditioned_alpha(seq_len=50, alpha_base=0.65, beta=0.3, ref_len=50)
    assert abs(alpha - 0.65) < 0.001, f"Expected 0.65, got {alpha}"


def test_length_conditioned_alpha_shorter():
    """At shorter seq_len, alpha should be larger."""
    alpha = length_conditioned_alpha(seq_len=30, alpha_base=0.65, beta=0.3, ref_len=50)
    assert alpha > 0.65, f"Expected >0.65, got {alpha}"


def test_length_conditioned_alpha_longer():
    """At longer seq_len, alpha should be smaller."""
    alpha = length_conditioned_alpha(seq_len=80, alpha_base=0.65, beta=0.3, ref_len=50)
    assert alpha < 0.65, f"Expected <0.65, got {alpha}"


def test_length_conditioned_alpha_clamped():
    """Alpha should be clamped to [0, 1]."""
    alpha_high = length_conditioned_alpha(seq_len=1, alpha_base=0.65, beta=10.0, ref_len=50)
    assert 0.0 <= alpha_high <= 1.0, f"Alpha {alpha_high} out of [0,1]"

    alpha_low = length_conditioned_alpha(seq_len=10000, alpha_base=0.65, beta=10.0, ref_len=50)
    assert 0.0 <= alpha_low <= 1.0, f"Alpha {alpha_low} out of [0,1]"


def test_simulate_spm_underfit_reproduces_baseline():
    """H1.2 baseline: alpha=0.65, seq_len=30 -> ratio≈1.21, seq_len=50 -> ratio≈1.12."""
    r30 = simulate_spm_underfit(30, 0.65, 6.93, 12.34)
    r50 = simulate_spm_underfit(50, 0.65, 7.81, 11.52)

    assert 1.20 <= r30["ratio"] <= 1.22, f"seq_len=30 ratio {r30['ratio']} not in [1.20, 1.22]"
    assert 1.11 <= r50["ratio"] <= 1.13, f"seq_len=50 ratio {r50['ratio']} not in [1.11, 1.13]"


def test_simulate_spm_underfit_monotonic():
    """Higher alpha should produce lower underfit (monotonic)."""
    prev = None
    for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        r = simulate_spm_underfit(30, alpha, 6.93, 12.34)
        if prev is not None:
            assert r["spm_underfit"] <= prev["spm_underfit"], \
                f"Non-monotonic at alpha={alpha}: {r['spm_underfit']} > {prev['spm_underfit']}"
        prev = r


def test_simulate_spm_underfit_bounds():
    """SPM underfit should be between CG and HM."""
    for seq_len in [30, 50]:
        for alpha in [0.0, 0.3, 0.65, 1.0]:
            cg = 6.93 if seq_len == 30 else 7.81
            hm = 12.34 if seq_len == 30 else 11.52
            r = simulate_spm_underfit(seq_len, alpha, cg, hm)
            # SPM should be <= HM (structural prior helps)
            assert r["spm_underfit"] <= hm + 0.01, \
                f"SPM underfit {r['spm_underfit']} > HM {hm}"
            # SPM can be slightly below CG at alpha=1.0 due to noise, but not by much
            assert r["spm_underfit"] >= cg - 0.5, \
                f"SPM underfit {r['spm_underfit']} << CG {cg}"


def test_run_experiment_returns_valid_structure():
    """Full experiment should return expected keys."""
    results = run_experiment()
    assert "experiment_id" in results
    assert results["experiment_id"] == "H1.2.1"
    assert "conclusion" in results
    assert results["conclusion"] in ("SUPPORTED", "REFUTED")
    assert "configurations" in results
    assert len(results["configurations"]) > 0
    assert "baseline_fixed_alpha" in results


def test_run_experiment_has_both_seq_lens():
    """All configurations should include both seq_len=30 and 50."""
    results = run_experiment()
    seq_lens = set(c["seq_len"] for c in results["configurations"])
    assert 30 in seq_lens, "Missing seq_len=30"
    assert 50 in seq_lens, "Missing seq_len=50"


def test_deterministic():
    """Two runs should produce identical results."""
    r1 = run_experiment()
    r2 = run_experiment()
    # Compare key metrics
    assert r1["conclusion"] == r2["conclusion"]
    assert r1.get("optimal_beta") == r2.get("optimal_beta")
    assert r1.get("minimal_beta") == r2.get("minimal_beta")
    # Compare all configurations
    for c1, c2 in zip(r1["configurations"], r2["configurations"]):
        assert c1["ratio"] == c2["ratio"], \
            f"Non-deterministic: {c1['ratio']} vs {c2['ratio']}"


if __name__ == "__main__":
    passed = 0
    failed = 0
    tests = [
        test_length_conditioned_alpha_baseline,
        test_length_conditioned_alpha_shorter,
        test_length_conditioned_alpha_longer,
        test_length_conditioned_alpha_clamped,
        test_simulate_spm_underfit_reproduces_baseline,
        test_simulate_spm_underfit_monotonic,
        test_simulate_spm_underfit_bounds,
        test_run_experiment_returns_valid_structure,
        test_run_experiment_has_both_seq_lens,
        test_deterministic,
    ]
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
