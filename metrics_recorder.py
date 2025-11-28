# -*- coding: utf-8 -*-
"""
Metrics Recording System for Laplace Enhancement Pipeline

Captures detailed metrics from each processing stage and saves as JSON/CSV
for downstream analysis and report generation.
"""

import json
import csv
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path


class MetricsRecorder:
    """
    Records and persists pipeline metrics for batch analysis

    Features:
    - JSON output for detailed hierarchical metrics
    - CSV output for tabular analysis (flat structure)
    - Append mode for batch processing
    - Summary statistics aggregation
    """

    def __init__(self, output_dir: str = "metrics"):
        """
        Initialize metrics recorder

        Args:
            output_dir: Directory to save metrics files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.json_path = self.output_dir / "pipeline_metrics.jsonl"  # JSON Lines format
        self.csv_path = self.output_dir / "pipeline_metrics.csv"
        self.summary_path = self.output_dir / "batch_summary.json"

        # Initialize CSV if needed
        self._initialize_csv()

    def _initialize_csv(self):
        """Create CSV file with headers if it doesn't exist"""
        if not self.csv_path.exists():
            headers = [
                # Identification
                'track_id',
                'track_name',
                'timestamp',

                # Input stats
                'audio_duration_s',
                'instruments_original',
                'notes_original',

                # Feature extraction
                'feature_extraction_time_s',
                'prony_success_rate_avg',
                'prony_success_rate_min',
                'prony_success_rate_max',
                'features_extracted',

                # Consolidation
                'consolidation_time_s',
                'consolidation_strategy',
                'instruments_after_consolidation',
                'instrument_pairs_merged',
                'instrument_reduction_count',
                'instrument_reduction_percent',
                'decay_threshold',
                'spectral_threshold',
                'fallback_mode',

                # Refinement
                'refinement_time_s',
                'refinement_status',
                'program_assignments',

                # Final output
                'instruments_final',
                'notes_final',
                'notes_reduction_count',
                'notes_reduction_percent',

                # Timing
                'total_time_s',
                'pipeline_status'
            ]

            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()

    def record_track_metrics(
        self,
        track_id: str,
        track_name: str,
        report: Dict[str, Any],
        audio_duration: float
    ):
        """
        Record metrics for a single track

        Args:
            track_id: Unique track identifier (e.g., "Track00001")
            track_name: Human-readable track name
            report: Enhancement report dict from Phase1Enhancer
            audio_duration: Audio duration in seconds
        """
        timestamp = datetime.now().isoformat()

        # Extract metrics from report
        metrics = self._extract_metrics(track_id, track_name, report, audio_duration, timestamp)

        # Save to JSON Lines (append)
        self._save_json(metrics)

        # Save to CSV (append)
        self._save_csv(metrics)

    def _extract_metrics(
        self,
        track_id: str,
        track_name: str,
        report: Dict[str, Any],
        audio_duration: float,
        timestamp: str
    ) -> Dict[str, Any]:
        """Extract flattened metrics from report dict"""

        # Feature extraction stage
        feature_stage = report.get('stages', {}).get('feature_extraction', {})
        feature_stats = feature_stage.get('stats', {})

        # Consolidation stage
        consolidation_stage = report.get('stages', {}).get('consolidation', {})
        consolidation_stats = consolidation_stage.get('stats', {})

        # Refinement stage
        refinement_stage = report.get('stages', {}).get('refinement', {})

        # Metrics stage
        metrics_stage = report.get('stages', {}).get('metrics', {})
        basic_metrics = metrics_stage.get('metrics', {})

        # Timing
        timing = report.get('timing', {})

        # Prony success rates per instrument
        prony_rates = feature_stats.get('prony_success_rates', [])
        prony_avg = sum(prony_rates) / len(prony_rates) if prony_rates else 0.0
        prony_min = min(prony_rates) if prony_rates else 0.0
        prony_max = max(prony_rates) if prony_rates else 0.0

        # Construct flat metrics dict
        metrics = {
            # Identification
            'track_id': track_id,
            'track_name': track_name,
            'timestamp': timestamp,

            # Input stats
            'audio_duration_s': round(audio_duration, 2),
            'instruments_original': basic_metrics.get('instruments_original', 0),
            'notes_original': basic_metrics.get('notes_original', 0),

            # Feature extraction
            'feature_extraction_time_s': round(timing.get('feature_extraction', 0), 2),
            'prony_success_rate_avg': round(prony_avg, 4),
            'prony_success_rate_min': round(prony_min, 4),
            'prony_success_rate_max': round(prony_max, 4),
            'features_extracted': feature_stats.get('instruments_processed', 0),

            # Consolidation
            'consolidation_time_s': round(timing.get('consolidation', 0), 2),
            'consolidation_strategy': consolidation_stats.get('strategy', 'unknown'),
            'instruments_after_consolidation': consolidation_stats.get('output_instruments', 0),
            'instrument_pairs_merged': consolidation_stats.get('merges_performed', 0),
            'instrument_reduction_count': consolidation_stats.get('input_instruments', 0) - consolidation_stats.get('output_instruments', 0),
            'instrument_reduction_percent': round(consolidation_stats.get('reduction_rate', 0) * 100, 2),
            'decay_threshold': consolidation_stats.get('decay_threshold', 0),
            'spectral_threshold': consolidation_stats.get('spectral_threshold', 0),
            'fallback_mode': consolidation_stats.get('fallback_mode', False),

            # Refinement
            'refinement_time_s': round(timing.get('refinement', 0), 2),
            'refinement_status': refinement_stage.get('status', 'unknown'),
            'program_assignments': len(refinement_stage.get('program_assignments', {})),

            # Final output
            'instruments_final': basic_metrics.get('instruments_enhanced', 0),
            'notes_final': basic_metrics.get('notes_enhanced', 0),
            'notes_reduction_count': basic_metrics.get('notes_original', 0) - basic_metrics.get('notes_enhanced', 0),
            'notes_reduction_percent': round(
                100 * (basic_metrics.get('notes_original', 0) - basic_metrics.get('notes_enhanced', 0)) / max(basic_metrics.get('notes_original', 1), 1),
                2
            ),

            # Timing
            'total_time_s': round(timing.get('total_seconds', 0), 2),
            'pipeline_status': report.get('pipeline_status', 'unknown')
        }

        # Add detailed hierarchical data for JSON
        metrics['_detailed'] = {
            'feature_stats': feature_stats,
            'consolidation_stats': consolidation_stats,
            'refinement_stage': refinement_stage,
            'merge_history': consolidation_stats.get('merge_history', [])
        }

        return metrics

    def _save_json(self, metrics: Dict[str, Any]):
        """Append metrics as JSON line"""
        with open(self.json_path, 'a') as f:
            json.dump(metrics, f, indent=None)
            f.write('\n')

    def _save_csv(self, metrics: Dict[str, Any]):
        """Append metrics to CSV (flat structure only)"""
        # Remove detailed hierarchical data for CSV
        flat_metrics = {k: v for k, v in metrics.items() if not k.startswith('_')}

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=flat_metrics.keys())
            writer.writerow(flat_metrics)

    def load_all_metrics(self) -> List[Dict[str, Any]]:
        """Load all recorded metrics from JSON Lines file"""
        if not self.json_path.exists():
            return []

        metrics_list = []
        with open(self.json_path, 'r') as f:
            for line in f:
                metrics_list.append(json.loads(line.strip()))

        return metrics_list

    def compute_batch_summary(self) -> Dict[str, Any]:
        """
        Compute summary statistics across all recorded tracks

        Returns:
            Dict with aggregated metrics and statistics
        """
        metrics_list = self.load_all_metrics()

        if not metrics_list:
            return {'status': 'no_data', 'total_tracks': 0}

        # Aggregate statistics
        total_tracks = len(metrics_list)
        successful = sum(1 for m in metrics_list if m['pipeline_status'] == 'success')
        failed = total_tracks - successful

        # Average metrics (successful tracks only)
        success_metrics = [m for m in metrics_list if m['pipeline_status'] == 'success']

        if not success_metrics:
            return {
                'status': 'all_failed',
                'total_tracks': total_tracks,
                'successful': 0,
                'failed': failed
            }

        def avg(key):
            values = [m[key] for m in success_metrics if key in m]
            return sum(values) / len(values) if values else 0

        summary = {
            'status': 'complete',
            'timestamp': datetime.now().isoformat(),
            'total_tracks': total_tracks,
            'successful': successful,
            'failed': failed,
            'success_rate': round(successful / total_tracks * 100, 2),

            # Average instrument reduction
            'avg_instruments_original': round(avg('instruments_original'), 1),
            'avg_instruments_final': round(avg('instruments_final'), 1),
            'avg_instrument_reduction_percent': round(avg('instrument_reduction_percent'), 2),
            'avg_instrument_pairs_merged': round(avg('instrument_pairs_merged'), 2),

            # Average Prony success
            'avg_prony_success_rate': round(avg('prony_success_rate_avg'), 4),

            # Average timing
            'avg_total_time_s': round(avg('total_time_s'), 2),
            'avg_feature_extraction_time_s': round(avg('feature_extraction_time_s'), 2),
            'avg_consolidation_time_s': round(avg('consolidation_time_s'), 2),
            'avg_refinement_time_s': round(avg('refinement_time_s'), 2),

            # Fallback mode usage
            'tracks_with_fallback': sum(1 for m in success_metrics if m.get('fallback_mode', False)),
            'fallback_rate': round(
                sum(1 for m in success_metrics if m.get('fallback_mode', False)) / len(success_metrics) * 100,
                2
            )
        }

        # Save summary
        with open(self.summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary

    def print_summary(self):
        """Print batch summary to console"""
        summary = self.compute_batch_summary()

        print("\n" + "=" * 70)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 70)

        if summary['status'] == 'no_data':
            print("No metrics recorded yet.")
            return

        print(f"\n📊 Overall Statistics:")
        print(f"  - Total tracks: {summary['total_tracks']}")
        print(f"  - Successful: {summary['successful']} ({summary['success_rate']:.1f}%)")
        print(f"  - Failed: {summary['failed']}")

        if summary['status'] == 'all_failed':
            return

        print(f"\n🎵 Instrument Reduction:")
        print(f"  - Average original: {summary['avg_instruments_original']:.1f}")
        print(f"  - Average final: {summary['avg_instruments_final']:.1f}")
        print(f"  - Average reduction: {summary['avg_instrument_reduction_percent']:.1f}%")
        print(f"  - Average pairs merged: {summary['avg_instrument_pairs_merged']:.2f}")

        print(f"\n📈 Prony Analysis:")
        print(f"  - Average success rate: {summary['avg_prony_success_rate']:.1%}")

        print(f"\n⏱️  Processing Time:")
        print(f"  - Average total: {summary['avg_total_time_s']:.1f}s ({summary['avg_total_time_s']/60:.1f} min)")
        print(f"  - Feature extraction: {summary['avg_feature_extraction_time_s']:.1f}s")
        print(f"  - Consolidation: {summary['avg_consolidation_time_s']:.1f}s")
        print(f"  - Refinement: {summary['avg_refinement_time_s']:.1f}s")

        print(f"\n⚠️  Fallback Mode:")
        print(f"  - Tracks using fallback: {summary['tracks_with_fallback']} ({summary['fallback_rate']:.1f}%)")

        print("\n" + "=" * 70)
        print(f"📁 Metrics saved to:")
        print(f"  - JSON: {self.json_path}")
        print(f"  - CSV: {self.csv_path}")
        print(f"  - Summary: {self.summary_path}")
        print("=" * 70)


if __name__ == "__main__":
    """Demo: Test metrics recording with synthetic data"""

    print("=" * 70)
    print("Metrics Recording Demo")
    print("=" * 70)

    # Create recorder
    recorder = MetricsRecorder(output_dir="demo_metrics")

    # Simulate processing 3 tracks
    for i in range(1, 4):
        track_id = f"Track{i:05d}"
        track_name = f"Test Track {i}"

        # Synthetic report
        report = {
            'pipeline_status': 'success',
            'stages': {
                'feature_extraction': {
                    'status': 'success',
                    'stats': {
                        'instruments_processed': 10 - i,
                        'prony_success_rates': [0.80, 0.75, 0.85, 0.70, 0.90] + [0.78] * (5 - i)
                    }
                },
                'consolidation': {
                    'status': 'success',
                    'stats': {
                        'input_instruments': 10 - i,
                        'output_instruments': 8 - i,
                        'merges_performed': 2,
                        'reduction_rate': 0.20,
                        'strategy': 'conservative',
                        'decay_threshold': 0.80,
                        'spectral_threshold': 200.0,
                        'fallback_mode': False,
                        'merge_history': [
                            {'iteration': 1, 'merged': (0, 1), 'decay_similarity': 0.85},
                            {'iteration': 2, 'merged': (2, 3), 'decay_similarity': 0.82}
                        ]
                    }
                },
                'refinement': {
                    'status': 'success',
                    'program_assignments': {0: 0, 1: 32, 2: 40}
                },
                'metrics': {
                    'status': 'success',
                    'metrics': {
                        'instruments_original': 10 - i,
                        'instruments_enhanced': 8 - i,
                        'notes_original': 1000 + i * 100,
                        'notes_enhanced': 1000 + i * 100  # Phase 1: no note reduction
                    }
                }
            },
            'timing': {
                'feature_extraction': 380.0 + i * 10,
                'consolidation': 5.0 + i * 0.5,
                'refinement': 2.0 + i * 0.2,
                'total_seconds': 400.0 + i * 15
            }
        }

        # Record metrics
        recorder.record_track_metrics(
            track_id=track_id,
            track_name=track_name,
            report=report,
            audio_duration=180.0 + i * 10
        )

        print(f"✓ Recorded metrics for {track_id}")

    # Print summary
    recorder.print_summary()

    print("\n--- Demo complete!")
