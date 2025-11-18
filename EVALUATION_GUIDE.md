# MR-MT3 + Laplace Enhancement Evaluation Guide

Comprehensive guide for collecting and analyzing results comparing baseline MR-MT3 vs Laplace-enhanced transcriptions.

---

## Quick Start

### Option 1: Integrated with Deployment Pipeline

Run the complete pipeline with automatic evaluation:

```bash
./scripts/deploy_mrmt3_pipeline.sh \
    --data-dir ./audio_input \
    --output-dir ./pipeline_output \
    --evaluate
```

This will:
1. Run MR-MT3 inference on all audio files
2. Apply Laplace enhancement to transcriptions
3. **Automatically generate comparison report**
4. Save HTML report to `./pipeline_output/evaluation/evaluation_report.html`

### Option 2: Standalone Evaluation

Run evaluation on existing outputs:

```bash
python evaluate_enhancement.py \
    --baseline-dir ./pipeline_output/mrmt3_transcriptions \
    --enhanced-dir ./pipeline_output/enhanced_transcriptions \
    --output-dir ./evaluation_results
```

---

## Output Structure

After running evaluation, you'll have:

```
evaluation_results/
├── evaluation_report.html          # Main visual report (open in browser)
├── detailed_results.json           # Raw metrics for all files
├── leakage_comparison.png          # Leakage reduction visualizations
└── accuracy_comparison.png         # Note accuracy charts (if ground truth provided)
```

---

## Understanding the Report

### 1. Summary Statistics

**Key Metrics Displayed:**
- **Total Files Evaluated**: Number of file pairs compared
- **Successful Comparisons**: Files processed without errors
- **Mean Leakage Reduction**: Average improvement in instrument separation

**Example:**
```
Total Files: 100
Successful: 98
Mean Leakage Reduction: 0.134 (13.4% improvement)
```

### 2. Instrument Leakage Reduction

**What is "Leakage"?**
- Leakage occurs when MR-MT3 assigns notes to wrong instruments
- Example: Bass notes appearing in piano track

**Metrics:**
- **Cross-Instrument Rate**: Fraction of notes appearing in wrong instruments
  - Lower = better
  - 0.2 means 20% of notes are leaked to wrong instruments

- **Temporal Overlap Ratio**: How much different instruments overlap in time
  - Lower = better separation
  - 0.5 means 50% of time multiple instruments play simultaneously

**Charts:**
1. **Box Plot**: Distribution comparison (baseline vs enhanced)
2. **Scatter Plot**: Per-file improvement (points below diagonal = improvement)
3. **Histogram**: Distribution of improvements across dataset
4. **Cumulative**: Percentage of files with given improvement level

### 3. Note Accuracy Improvement (with Ground Truth)

Only available if you provide ground truth MIDI files.

**Metrics:**
- **Precision**: Of detected notes, how many are correct?
  - Higher = fewer false positives

- **Recall**: Of ground truth notes, how many were detected?
  - Higher = fewer missed notes

- **F1 Score**: Harmonic mean of precision and recall
  - Balanced metric (0.0 to 1.0, higher is better)

**Example Interpretation:**
```
Baseline F1: 0.723
Enhanced F1: 0.781
Improvement: +0.058 (+8.0%)
```
This means Laplace enhancement improved overall note accuracy by 8%.

---

## Advanced Usage

### With Ground Truth (for Slakh2100)

If you have ground truth MIDI files, add them for accuracy metrics:

```bash
python evaluate_enhancement.py \
    --baseline-dir ./mrmt3_outputs \
    --enhanced-dir ./enhanced_outputs \
    --ground-truth-dir ./slakh2100/Track*/MIDI \
    --output-dir ./evaluation_results
```

### With Audio Files (for Context)

Include original audio for reference:

```bash
python evaluate_enhancement.py \
    --baseline-dir ./mrmt3_outputs \
    --enhanced-dir ./enhanced_outputs \
    --audio-dir ./audio_files \
    --output-dir ./evaluation_results
```

### Batch Evaluation for Research

Process multiple experiments:

```bash
# Experiment 1: Default parameters
python evaluate_enhancement.py \
    --baseline-dir ./exp1/baseline \
    --enhanced-dir ./exp1/enhanced \
    --output-dir ./exp1/results

# Experiment 2: Different thresholds
python evaluate_enhancement.py \
    --baseline-dir ./exp2/baseline \
    --enhanced-dir ./exp2/enhanced \
    --output-dir ./exp2/results

# Compare experiments by opening both HTML reports
```

---

## Interpreting Results

### What Indicates Success?

**Good Results:**
- ✅ Mean leakage reduction > 0.1 (10% improvement)
- ✅ Most files show positive improvement (>60%)
- ✅ F1 score improvement > 0.05 (5% improvement)
- ✅ Low variance (consistent across different files)

**Needs Tuning:**
- ⚠️ Mean leakage reduction < 0.05 (5% improvement)
- ⚠️ High variance (works well for some files, poorly for others)
- ⚠️ Negative improvements for many files

**Configuration Issues:**
- ❌ Mean leakage reduction < 0 (making things worse)
- ❌ Most files show negative improvement
- ❌ F1 score decreased

### File-Specific Analysis

In `detailed_results.json`, find per-file metrics:

```json
{
  "filename": "Track00042",
  "baseline": {
    "leakage": {
      "cross_instrument_rate": 0.234,
      "temporal_overlap_ratio": 0.512
    }
  },
  "enhanced": {
    "leakage": {
      "cross_instrument_rate": 0.087,
      "temporal_overlap_ratio": 0.301
    }
  },
  "improvement": {
    "leakage_reduction": {
      "cross_instrument_rate": 0.147,    // 14.7% reduction
      "temporal_overlap_ratio": 0.211    // 21.1% reduction
    }
  }
}
```

**Interpreting This File:**
- Strong improvement in instrument separation (14.7% fewer leaked notes)
- Good reduction in temporal overlap (better time separation)
- This file benefited significantly from Laplace enhancement

### Common Patterns

**When Enhancement Works Best:**
- Multi-instrument tracks with similar frequency ranges
- Songs with clear attack/decay differences between instruments
- Tracks with moderate polyphony (3-6 simultaneous instruments)

**When Enhancement May Struggle:**
- Single instrument tracks (nothing to separate)
- Heavily processed audio (reverb/effects blur decay characteristics)
- Very dense polyphony (>10 simultaneous instruments)

---

## Exporting Results for Papers/Presentations

### Generate Publication-Quality Figures

The PNG files are 300 DPI and suitable for papers:
- `leakage_comparison.png`: 4-panel leakage analysis
- `accuracy_comparison.png`: Precision/Recall/F1 comparison

### Extract Statistical Summaries

From `detailed_results.json`:

```python
import json

with open('evaluation_results/detailed_results.json') as f:
    results = json.load(f)

# Calculate statistics for paper
baseline_leakage = [r['leakage']['cross_instrument_rate'] for r in results['baseline']]
enhanced_leakage = [r['leakage']['cross_instrument_rate'] for r in results['enhanced']]

import numpy as np
print(f"Baseline: {np.mean(baseline_leakage):.3f} ± {np.std(baseline_leakage):.3f}")
print(f"Enhanced: {np.mean(enhanced_leakage):.3f} ± {np.std(enhanced_leakage):.3f}")
```

### Key Numbers for Reporting

From HTML report, extract:
1. **Sample size**: "Total Files Evaluated"
2. **Primary outcome**: "Mean Leakage Reduction"
3. **Effect size**: Improvement percentage
4. **Variance**: Standard deviation (from detailed_results.json)
5. **Secondary outcomes**: F1 improvement, precision/recall changes

**Example Paper Text:**
> We evaluated the Laplace enhancement pipeline on 100 tracks from Slakh2100.
> Compared to baseline MR-MT3, the enhanced pipeline reduced cross-instrument
> leakage by 13.4% on average (σ = 0.089), with 87% of tracks showing improvement.
> Note accuracy (F1 score) improved by 8.0% (baseline: 0.723, enhanced: 0.781).

---

## Troubleshooting

### Error: "No enhanced version found for X"

**Cause**: Filename mismatch between baseline and enhanced directories

**Solution**: Enhanced files should be named either:
- Same as baseline: `Track00001.mid`
- With suffix: `Track00001_enhanced.mid`

### Error: "ModuleNotFoundError: No module named 'laplace_mrmt3'"

**Cause**: Python can't find the enhancement module

**Solution**:
```bash
# Make sure you're in the research directory
cd /path/to/Laplace/research

# Install in development mode
pip install -e .
```

### Warning: "No ground truth for X"

**Cause**: Ground truth file not found (optional feature)

**Impact**: Accuracy metrics won't be available, but leakage metrics still work

**Solution**: Either provide ground truth files or ignore (leakage metrics are primary)

### Charts Not Displaying in HTML Report

**Cause**: PNG files not generated or in wrong location

**Solution**:
```bash
# Check that PNG files exist
ls evaluation_results/*.png

# If missing, check evaluation logs for matplotlib errors
cat pipeline_output/logs/evaluation.log
```

---

## Integration with CI/CD

### Automated Testing

Add to your testing pipeline:

```bash
#!/bin/bash
# Run pipeline with test samples
./scripts/deploy_mrmt3_pipeline.sh \
    --data-dir ./test_samples \
    --output-dir ./test_output \
    --test-mode \
    --evaluate

# Check that leakage reduction meets threshold
python -c "
import json
with open('./test_output/evaluation/detailed_results.json') as f:
    data = json.load(f)

baseline = sum(r['leakage']['cross_instrument_rate'] for r in data['baseline']) / len(data['baseline'])
enhanced = sum(r['leakage']['cross_instrument_rate'] for r in data['enhanced']) / len(data['enhanced'])
improvement = baseline - enhanced

assert improvement > 0.05, f'Insufficient improvement: {improvement:.3f}'
print(f'✓ Leakage reduction: {improvement:.3f}')
"
```

### Performance Regression Detection

Track metrics over time:

```bash
# Save baseline metrics
cp evaluation_results/detailed_results.json metrics_baseline_v1.0.json

# After code changes, compare
python compare_versions.py metrics_baseline_v1.0.json evaluation_results/detailed_results.json
```

---

## Reference: Complete Evaluation Workflow

For complete research evaluation:

```bash
# 1. Download dataset
wget https://zenodo.org/records/4599666/files/babyslakh_16k.tar.gz
tar -xzf babyslakh_16k.tar.gz

# 2. Run complete pipeline with evaluation
./scripts/deploy_mrmt3_pipeline.sh \
    --data-dir ./babyslakh_16k \
    --output-dir ./results \
    --batch-size 8 \
    --workers 4 \
    --evaluate

# 3. Open report
open ./results/evaluation/evaluation_report.html

# 4. Extract statistics for publication
python -c "
import json
import numpy as np

with open('./results/evaluation/detailed_results.json') as f:
    data = json.load(f)

baseline_leakage = [r['leakage']['cross_instrument_rate'] for r in data['baseline']]
enhanced_leakage = [r['leakage']['cross_instrument_rate'] for r in data['enhanced']]

print(f'Baseline leakage: {np.mean(baseline_leakage):.4f} ± {np.std(baseline_leakage):.4f}')
print(f'Enhanced leakage: {np.mean(enhanced_leakage):.4f} ± {np.std(enhanced_leakage):.4f}')
print(f'Mean reduction: {np.mean(baseline_leakage) - np.mean(enhanced_leakage):.4f}')
print(f'Median reduction: {np.median(baseline_leakage) - np.median(enhanced_leakage):.4f}')

improvements = np.array(baseline_leakage) - np.array(enhanced_leakage)
print(f'Files improved: {np.sum(improvements > 0) / len(improvements) * 100:.1f}%')
"

# 5. Archive results
tar -czf evaluation_$(date +%Y%m%d).tar.gz results/evaluation/
```

---

## Contact

For issues with evaluation framework, check:
- Metrics implementation: `laplace_mrmt3/metrics.py`
- Evaluation script: `evaluate_enhancement.py`
- Deployment integration: `scripts/deploy_mrmt3_pipeline.sh`
