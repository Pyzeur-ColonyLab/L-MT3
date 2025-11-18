# Metrics Module Usage Guide

## Overview

The `metrics.py` module implements three key metrics for Phase 1 MR-MT3 Laplace Enhancement evaluation:

1. **Instrument Leakage Ratio (φ)** - Measures over-prediction of instruments
2. **Decay Consistency Score** - Measures variance of exponential decay rates
3. **Timbre Homogeneity** - Measures spectral variance within instruments

All metrics include:
- Statistical rigor (95% confidence intervals)
- Bootstrap resampling for robust estimation
- Human-readable interpretation
- Target achievement assessment

## Quick Start

```python
from laplace_mrmt3.metrics import EnhancementMetrics, compare_transcriptions
from laplace_mrmt3.config import MetricsConfig
import numpy as np

# Initialize metrics calculator
metrics = EnhancementMetrics()

# Example data
predicted_instruments = [0, 24, 40, 42]  # MIDI program numbers
ground_truth_instruments = [0, 24, 40]   # True instruments
decay_rates = np.random.normal(0.5, 0.08, 100)
spectral_centroids = np.random.normal(800.0, 80.0, 100)

# Compute all metrics
report = metrics.evaluate_enhancement(
    predicted_instruments=predicted_instruments,
    ground_truth_instruments=ground_truth_instruments,
    decay_rates=decay_rates,
    spectral_centroids=spectral_centroids
)

# Print report
metrics.print_report(report)
```

## Metric Formulas

### 1. Instrument Leakage Ratio (φ)

```
φ = n_predicted_instruments / n_true_instruments
```

**Interpretation:**
- φ = 1.0: Perfect prediction
- φ > 1.0: Over-prediction (instrument leakage)
- φ < 1.0: Under-prediction (missed instruments)

**Targets:**
- Baseline (MR-MT3): φ = 1.24
- Minimum: φ < 1.18 (>5% improvement)
- Goal: φ < 1.12 (>10% improvement)
- Excellent: φ < 1.05 (>15% improvement)

### 2. Decay Consistency Score

```
cv = std(decay_rates) / |mean(decay_rates)|  # Coefficient of variation
consistency = 1 / (1 + cv)
```

**Interpretation:**
- consistency = 1.0: Perfect (no variance)
- consistency → 0: High variance

**Targets:**
- Minimum: > 0.65
- Goal: > 0.75
- Excellent: > 0.80

### 3. Timbre Homogeneity

```
homogeneity = 1 - (std(centroids) / mean(centroids))
```

**Interpretation:**
- homogeneity = 1.0: Perfect (no variance)
- homogeneity → 0: High variance

**Targets:**
- Baseline (estimated): 0.55
- Goal: > 0.75

## Phase 1 Success Criteria

Phase 1 is considered successful if:
1. **Leakage ratio meets target** (φ < 1.12), AND
2. **At least 2 out of 3 targets are met**

Weighted score formula:
```
overall_score = 0.5 * leakage_score + 0.25 * decay_score + 0.25 * timbre_score
```

## Advanced Usage

### Per-Instrument Metrics

```python
# Compute metrics separately for each instrument
instrument_labels_decay = [0, 0, 0, 24, 24, 40, 40, 40, ...]
instrument_labels_spectral = [0, 0, 24, 24, 40, 40, ...]

report = metrics.evaluate_enhancement(
    predicted_instruments=predicted,
    ground_truth_instruments=ground_truth,
    decay_rates=decay_rates,
    spectral_centroids=spectral_centroids,
    instrument_labels_decay=instrument_labels_decay,
    instrument_labels_spectral=instrument_labels_spectral
)
```

### Comparing Vanilla vs Enhanced

```python
from laplace_mrmt3.metrics import compare_transcriptions

report = compare_transcriptions(
    vanilla_midi_path="outputs/vanilla_transcription.mid",
    enhanced_midi_path="outputs/enhanced_transcription.mid",
    ground_truth_midi_path="data/ground_truth.mid",
    decay_rates=decay_rates,
    spectral_centroids=spectral_centroids
)
```

### Per-Instrument Analysis

```python
from laplace_mrmt3.metrics import compute_per_instrument_metrics

# Decay rates and centroids organized by instrument program
decay_by_instrument = {
    0: [0.5, 0.48, 0.52, ...],   # Piano
    24: [0.6, 0.58, 0.62, ...],  # Guitar
    40: [0.3, 0.29, 0.31, ...]   # Strings
}

spectral_by_instrument = {
    0: [800, 820, 790, ...],
    24: [1200, 1180, 1220, ...],
    40: [600, 610, 590, ...]
}

per_inst_metrics = compute_per_instrument_metrics(
    midi_path="outputs/transcription.mid",
    decay_rates_per_note=decay_by_instrument,
    spectral_centroids_per_note=spectral_by_instrument
)

# Access individual instrument metrics
piano_metrics = per_inst_metrics[0]
print(f"Piano decay consistency: {piano_metrics['decay_consistency'].value:.4f}")
```

## Custom Configuration

```python
from laplace_mrmt3.config import MetricsConfig

# Custom targets
custom_config = MetricsConfig(
    phi_target_goal=1.10,  # Stricter leakage target
    decay_consistency_goal=0.80,  # Higher consistency target
    timbre_homogeneity_goal=0.80   # Higher homogeneity target
)

metrics = EnhancementMetrics(config=custom_config)
```

## Output Interpretation

### Example Output

```
======================================================================
PHASE 1 ENHANCEMENT EVALUATION REPORT
======================================================================

1. INSTRUMENT LEAKAGE RATIO (phi)
   Value: 1.0800 (baseline: 1.24)
   95% CI: [1.0500, 1.1100]
   Improvement: 66.7%
   Target met: YES
   Interpretation: Excellent: 66.7% improvement over baseline

2. DECAY CONSISTENCY
   Value: 0.8200
   95% CI: [0.8000, 0.8400]
   Target met: YES
   Interpretation: Excellent: very consistent decay patterns

3. TIMBRE HOMOGENEITY
   Value: 0.7800
   95% CI: [0.7600, 0.8000]
   Target met: YES
   Interpretation: Good: meets target homogeneity

OVERALL PHASE 1 SUCCESS: PASS
Overall improvement score: 0.82
======================================================================
```

### Reading the Report

- **95% CI**: Confidence interval - the true value likely falls within this range
- **Improvement percentage**: How much of the gap to perfect has been closed
- **Target met**: Whether the metric achieves Phase 1 goals
- **Interpretation**: Human-readable assessment of the metric value

## Statistical Methods

### Bootstrap Confidence Intervals
- 1000 resamples for robust estimation
- Percentile method (2.5th and 97.5th percentiles)
- Handles small sample sizes and non-normal distributions

### Standard Error Computation
- Used for per-instrument aggregation
- Normal approximation for large samples
- Provides measure of estimation uncertainty

## Integration with Pipeline

```python
# Typical workflow
from laplace_mrmt3.features import FeatureExtractor
from laplace_mrmt3.metrics import EnhancementMetrics

# 1. Extract features
extractor = FeatureExtractor(config)
features = extractor.extract_from_midi(midi_path, audio_path)

# 2. Compute metrics
metrics = EnhancementMetrics()
report = metrics.evaluate_enhancement(
    predicted_instruments=features['instrument_programs'],
    ground_truth_instruments=ground_truth,
    decay_rates=features['decay_rates'],
    spectral_centroids=features['spectral_centroids']
)

# 3. Evaluate success
if report.phase1_success:
    print("Phase 1 enhancement successful!")
else:
    print(f"Phase 1 incomplete. Overall score: {report.overall_improvement:.2f}")
```

## Validation Results

The validation test demonstrates three scenarios:

1. **Perfect Separation**: φ=1.0, consistency=0.98, homogeneity=0.98 → PASS
2. **Baseline Performance**: φ=1.67, consistency=0.76, homogeneity=0.81 → FAIL (leakage too high)
3. **Target Performance**: φ=1.33, consistency=0.86, homogeneity=0.89 → FAIL (leakage still high)

This shows that **leakage reduction is the critical bottleneck** for Phase 1 success.

## References

- Section 6 of PHASE1_MRMT3_SPECIFICATION.md
- Appendix A: Parameter Reference
- MR-MT3 baseline: Gardner et al. (2023)
