# Metrics Recording System - Usage Guide

## Overview

The metrics recording system captures detailed processing statistics from each track in the Laplace enhancement pipeline, saving them in both JSON and CSV formats for easy analysis and report generation.

## Features

- **JSON Lines format**: Detailed hierarchical metrics with full report data
- **CSV format**: Flat tabular data for spreadsheet analysis
- **Batch summary**: Aggregated statistics across all processed tracks
- **Resume-safe**: Appends metrics, safe for interrupted batch runs

## Quick Start

### Single Track with Metrics

```bash
cd ~/L-MT3

python3 research/phase1_mrmt3_enhancement.py \
  --midi ~/L-MT3/babyslakh_16k/Track00001/Track00001_baseline.mid \
  --audio ~/L-MT3/babyslakh_16k/Track00001/mix.wav \
  --metrics-dir ~/L-MT3/metrics
```

### Batch Processing with Metrics

```bash
cd ~/L-MT3

# Run batch with metrics recording
chmod +x run_batch_pipeline_with_metrics.sh
./run_batch_pipeline_with_metrics.sh
```

### View Summary

```bash
cd ~/L-MT3

python3 research/show_metrics_summary.py --metrics-dir ~/L-MT3/metrics
```

## Output Files

### Directory Structure
```
~/L-MT3/metrics/
├── pipeline_metrics.jsonl    # JSON Lines (one JSON object per line)
├── pipeline_metrics.csv       # CSV table
└── batch_summary.json         # Aggregated statistics
```

### Metrics Captured

#### Identification
- `track_id`: Track identifier (e.g., Track00001)
- `track_name`: Full track name
- `timestamp`: Processing timestamp

#### Input Statistics
- `audio_duration_s`: Audio duration in seconds
- `instruments_original`: Number of instruments before processing
- `notes_original`: Total notes before processing

#### Feature Extraction
- `feature_extraction_time_s`: Processing time
- `prony_success_rate_avg`: Average Prony success rate
- `prony_success_rate_min`: Minimum Prony success rate
- `prony_success_rate_max`: Maximum Prony success rate
- `features_extracted`: Number of instruments analyzed

#### Consolidation
- `consolidation_time_s`: Processing time
- `consolidation_strategy`: Strategy used (conservative/balanced/aggressive)
- `instruments_after_consolidation`: Instruments after consolidation
- `instrument_pairs_merged`: Number of instrument pairs merged
- `instrument_reduction_count`: Absolute reduction
- `instrument_reduction_percent`: Percentage reduction
- `decay_threshold`: Decay similarity threshold used
- `spectral_threshold`: Spectral difference threshold used
- `fallback_mode`: Whether Prony fallback was triggered

#### Refinement
- `refinement_time_s`: Processing time
- `refinement_status`: Success/failure status
- `program_assignments`: Number of GM program assignments

#### Final Output
- `instruments_final`: Final instrument count
- `notes_final`: Final note count
- `notes_reduction_count`: Absolute note reduction
- `notes_reduction_percent`: Percentage note reduction

#### Timing
- `total_time_s`: Total processing time
- `pipeline_status`: Overall status (success/failed)

## Using Metrics for Analysis

### Load in Python

```python
import json
import pandas as pd

# Load JSON Lines
metrics_list = []
with open('metrics/pipeline_metrics.jsonl', 'r') as f:
    for line in f:
        metrics_list.append(json.loads(line))

# Load CSV
df = pd.read_csv('metrics/pipeline_metrics.csv')

# Analyze
print(f"Average instrument reduction: {df['instrument_reduction_percent'].mean():.1f}%")
print(f"Average Prony success: {df['prony_success_rate_avg'].mean():.1%}")
```

### Load in R

```r
# Load CSV
metrics <- read.csv('metrics/pipeline_metrics.csv')

# Summary statistics
summary(metrics$instrument_reduction_percent)
summary(metrics$prony_success_rate_avg)

# Visualization
library(ggplot2)

ggplot(metrics, aes(x=prony_success_rate_avg, y=instrument_reduction_percent)) +
  geom_point() +
  labs(title="Prony Success vs Instrument Reduction",
       x="Prony Success Rate", y="Instrument Reduction %")
```

### Analyze in Spreadsheet

1. Open `metrics/pipeline_metrics.csv` in Excel/Google Sheets
2. Create pivot tables for analysis
3. Generate charts and reports

## Batch Summary Format

The `batch_summary.json` file contains aggregated statistics:

```json
{
  "status": "complete",
  "timestamp": "2025-11-25T14:30:00",
  "total_tracks": 100,
  "successful": 98,
  "failed": 2,
  "success_rate": 98.0,

  "avg_instruments_original": 10.2,
  "avg_instruments_final": 8.5,
  "avg_instrument_reduction_percent": 16.7,
  "avg_instrument_pairs_merged": 1.7,

  "avg_prony_success_rate": 0.733,

  "avg_total_time_s": 412.3,
  "avg_feature_extraction_time_s": 398.1,
  "avg_consolidation_time_s": 8.2,
  "avg_refinement_time_s": 6.0,

  "tracks_with_fallback": 5,
  "fallback_rate": 5.1
}
```

## Advanced Usage

### Custom Metrics Directory

```bash
python3 research/phase1_mrmt3_enhancement.py \
  --midi input.mid \
  --audio input.wav \
  --metrics-dir /custom/path/to/metrics
```

### Programmatic Access

```python
from research.metrics_recorder import MetricsRecorder

# Create recorder
recorder = MetricsRecorder(output_dir="metrics")

# After processing track
recorder.record_track_metrics(
    track_id="Track00001",
    track_name="Track00001_mix",
    report=enhancement_report,
    audio_duration=180.5
)

# Generate summary
summary = recorder.compute_batch_summary()
recorder.print_summary()
```

### Metrics-Only Mode (Without Reprocessing)

If you have existing reports, you can record metrics without reprocessing:

```python
from research.metrics_recorder import MetricsRecorder
import json

recorder = MetricsRecorder(output_dir="metrics")

# Load existing report
with open('track_report.json', 'r') as f:
    report = json.load(f)

# Record metrics
recorder.record_track_metrics(
    track_id="Track00001",
    track_name="Track00001",
    report=report,
    audio_duration=180.5
)
```

## Troubleshooting

### No metrics files generated
- Check `--metrics-dir` parameter is provided
- Verify write permissions on metrics directory
- Check logs for "Metrics recording enabled" message

### Missing tracks in summary
- Tracks are only recorded on successful completion
- Check individual track logs for errors
- Failed tracks won't appear in metrics

### Inconsistent metrics
- Ensure using same pipeline version across batch
- Check for interrupted runs (use --clean for fresh start)
- Verify all tracks use same configuration

## Integration with Reports

Metrics can be used to generate reports:

```python
from research.metrics_recorder import MetricsRecorder
import matplotlib.pyplot as plt

recorder = MetricsRecorder(output_dir="metrics")
metrics_list = recorder.load_all_metrics()

# Extract data
reduction_rates = [m['instrument_reduction_percent'] for m in metrics_list]
prony_rates = [m['prony_success_rate_avg'] for m in metrics_list]

# Generate report visualization
plt.figure(figsize=(10, 6))
plt.scatter(prony_rates, reduction_rates, alpha=0.6)
plt.xlabel('Prony Success Rate')
plt.ylabel('Instrument Reduction %')
plt.title('Phase 1 Results: Prony Success vs Instrument Reduction')
plt.grid(True, alpha=0.3)
plt.savefig('phase1_analysis.png', dpi=300)
```

## Next Steps

1. **Run batch processing**: `./run_batch_pipeline_with_metrics.sh`
2. **View summary**: `python3 research/show_metrics_summary.py`
3. **Analyze CSV**: Open in Excel/Sheets for detailed analysis
4. **Generate reports**: Use metrics for Phase 1 validation report
