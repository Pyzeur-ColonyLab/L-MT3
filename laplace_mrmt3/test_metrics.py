#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Standalone test script for metrics module"""

import sys
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, '/Volumes/T7/Dyapason/Fourier_Laplace/Laplace/research')

from laplace_mrmt3.metrics import EnhancementMetrics, MetricResult, EnhancementReport
from laplace_mrmt3.config import MetricsConfig

def test_metrics():
    """Test all three metrics with synthetic data"""

    print("=" * 70)
    print("PHASE 1 METRICS MODULE - VALIDATION")
    print("=" * 70)

    # Create sample data
    rng = np.random.default_rng(42)

    # Scenario 1: Perfect separation (ideal case)
    print("\nScenario 1: Perfect Separation")
    print("-" * 70)

    ground_truth = [0, 24, 40]  # Piano, Guitar, Strings
    predicted_perfect = [0, 24, 40]  # Exact match
    decay_perfect = rng.normal(loc=0.5, scale=0.01, size=100)  # Very low variance
    spectral_perfect = rng.normal(loc=800.0, scale=20.0, size=100)  # Very low variance

    metrics = EnhancementMetrics()
    report1 = metrics.evaluate_enhancement(
        predicted_perfect, ground_truth,
        decay_perfect, spectral_perfect
    )
    metrics.print_report(report1)

    # Scenario 2: Baseline MR-MT3 (expected leakage)
    print("\n\nScenario 2: Baseline MR-MT3 Performance")
    print("-" * 70)

    predicted_baseline = [0, 1, 24, 25, 40]  # Over-prediction (leakage)
    decay_baseline = rng.normal(loc=0.5, scale=0.15, size=100)  # Higher variance
    spectral_baseline = rng.normal(loc=800.0, scale=150.0, size=100)  # Higher variance

    report2 = metrics.evaluate_enhancement(
        predicted_baseline, ground_truth,
        decay_baseline, spectral_baseline
    )
    metrics.print_report(report2)

    # Scenario 3: Phase 1 Target Performance
    print("\n\nScenario 3: Phase 1 Target Performance")
    print("-" * 70)

    predicted_target = [0, 24, 40, 42]  # Small leakage (phi = 1.33)
    decay_target = rng.normal(loc=0.5, scale=0.08, size=100)  # Moderate variance
    spectral_target = rng.normal(loc=800.0, scale=80.0, size=100)  # Moderate variance

    report3 = metrics.evaluate_enhancement(
        predicted_target, ground_truth,
        decay_target, spectral_target
    )
    metrics.print_report(report3)

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    # Summary of tests
    print("\nTEST SUMMARY:")
    print(f"  Scenario 1 (Perfect): Phase 1 Success = {report1.phase1_success}")
    print(f"  Scenario 2 (Baseline): Phase 1 Success = {report2.phase1_success}")
    print(f"  Scenario 3 (Target): Phase 1 Success = {report3.phase1_success}")

    return report1, report2, report3


if __name__ == "__main__":
    test_metrics()
