#!/usr/bin/env python3
"""
Evaluation Framework for MR-MT3 + Laplace Enhancement
Compares baseline MR-MT3 outputs vs Laplace-enhanced outputs

Usage:
    python evaluate_enhancement.py --baseline-dir ./mrmt3_outputs \
                                    --enhanced-dir ./enhanced_outputs \
                                    --ground-truth-dir ./ground_truth \
                                    --output-dir ./evaluation_results
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server deployment

from laplace_mrmt3.metrics import (
    InstrumentLeakageMetrics,
    NoteAccuracyMetrics,
    TimingMetrics,
    VelocityMetrics
)

import pretty_midi


class EnhancementEvaluator:
    """Compare baseline MR-MT3 vs Laplace-enhanced transcriptions"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results = {
            'baseline': [],
            'enhanced': [],
            'metadata': {
                'total_files': 0,
                'successful_comparisons': 0,
                'failed_comparisons': 0
            }
        }

    def evaluate_file_pair(
        self,
        baseline_midi: Path,
        enhanced_midi: Path,
        ground_truth_midi: Path = None,
        audio_path: Path = None
    ) -> Dict:
        """Evaluate single file pair"""

        try:
            baseline_pm = pretty_midi.PrettyMIDI(str(baseline_midi))
            enhanced_pm = pretty_midi.PrettyMIDI(str(enhanced_midi))

            # Load ground truth if available
            gt_pm = None
            if ground_truth_midi and ground_truth_midi.exists():
                gt_pm = pretty_midi.PrettyMIDI(str(ground_truth_midi))

            file_results = {
                'filename': baseline_midi.stem,
                'baseline': {},
                'enhanced': {},
                'improvement': {}
            }

            # 1. Instrument Leakage Metrics (primary goal)
            leakage_metrics = InstrumentLeakageMetrics()

            baseline_leakage = leakage_metrics.calculate(baseline_pm)
            enhanced_leakage = leakage_metrics.calculate(enhanced_pm)

            file_results['baseline']['leakage'] = baseline_leakage
            file_results['enhanced']['leakage'] = enhanced_leakage
            file_results['improvement']['leakage_reduction'] = {
                'cross_instrument_rate':
                    baseline_leakage['cross_instrument_rate'] - enhanced_leakage['cross_instrument_rate'],
                'temporal_overlap_ratio':
                    baseline_leakage['temporal_overlap_ratio'] - enhanced_leakage['temporal_overlap_ratio']
            }

            # 2. Note Accuracy Metrics (if ground truth available)
            if gt_pm:
                note_metrics = NoteAccuracyMetrics()

                # Compare baseline vs ground truth
                baseline_accuracy = note_metrics.calculate(baseline_pm, gt_pm)
                file_results['baseline']['accuracy'] = baseline_accuracy

                # Compare enhanced vs ground truth
                enhanced_accuracy = note_metrics.calculate(enhanced_pm, gt_pm)
                file_results['enhanced']['accuracy'] = enhanced_accuracy

                # Calculate improvement
                file_results['improvement']['accuracy'] = {
                    'precision_delta': enhanced_accuracy['precision'] - baseline_accuracy['precision'],
                    'recall_delta': enhanced_accuracy['recall'] - baseline_accuracy['recall'],
                    'f1_delta': enhanced_accuracy['f1_score'] - baseline_accuracy['f1_score']
                }

            # 3. Timing Metrics
            timing_metrics = TimingMetrics()

            baseline_timing = timing_metrics.calculate(baseline_pm, gt_pm) if gt_pm else timing_metrics.calculate(baseline_pm)
            enhanced_timing = timing_metrics.calculate(enhanced_pm, gt_pm) if gt_pm else timing_metrics.calculate(enhanced_pm)

            file_results['baseline']['timing'] = baseline_timing
            file_results['enhanced']['timing'] = enhanced_timing

            # 4. Velocity Metrics
            velocity_metrics = VelocityMetrics()

            baseline_velocity = velocity_metrics.calculate(baseline_pm, gt_pm) if gt_pm else velocity_metrics.calculate(baseline_pm)
            enhanced_velocity = velocity_metrics.calculate(enhanced_pm, gt_pm) if gt_pm else velocity_metrics.calculate(enhanced_pm)

            file_results['baseline']['velocity'] = baseline_velocity
            file_results['enhanced']['velocity'] = enhanced_velocity

            return file_results

        except Exception as e:
            print(f"Error evaluating {baseline_midi.stem}: {e}")
            return None

    def evaluate_dataset(
        self,
        baseline_dir: Path,
        enhanced_dir: Path,
        ground_truth_dir: Path = None,
        audio_dir: Path = None
    ):
        """Evaluate entire dataset"""

        baseline_files = sorted(Path(baseline_dir).glob("*.mid"))

        print(f"Found {len(baseline_files)} baseline MIDI files")

        for baseline_file in baseline_files:
            # Find corresponding enhanced file
            enhanced_file = Path(enhanced_dir) / f"{baseline_file.stem}_enhanced.mid"
            if not enhanced_file.exists():
                enhanced_file = Path(enhanced_dir) / baseline_file.name

            if not enhanced_file.exists():
                print(f"Warning: No enhanced version for {baseline_file.stem}")
                self.results['metadata']['failed_comparisons'] += 1
                continue

            # Find ground truth if available
            gt_file = None
            if ground_truth_dir:
                gt_file = Path(ground_truth_dir) / baseline_file.name
                if not gt_file.exists():
                    gt_file = None

            # Find audio if available
            audio_file = None
            if audio_dir:
                for ext in ['.wav', '.mp3', '.flac']:
                    audio_file = Path(audio_dir) / f"{baseline_file.stem}{ext}"
                    if audio_file.exists():
                        break
                else:
                    audio_file = None

            # Evaluate pair
            print(f"Evaluating: {baseline_file.stem}")
            result = self.evaluate_file_pair(
                baseline_file,
                enhanced_file,
                gt_file,
                audio_file
            )

            if result:
                self.results['baseline'].append(result['baseline'])
                self.results['enhanced'].append(result['enhanced'])
                self.results['metadata']['successful_comparisons'] += 1
            else:
                self.results['metadata']['failed_comparisons'] += 1

        self.results['metadata']['total_files'] = len(baseline_files)

    def generate_summary_statistics(self) -> Dict:
        """Calculate aggregate statistics across dataset"""

        summary = {
            'leakage_reduction': {},
            'accuracy_improvement': {},
            'timing_comparison': {},
            'velocity_comparison': {}
        }

        # Aggregate leakage metrics
        baseline_cross_inst = [r['leakage']['cross_instrument_rate'] for r in self.results['baseline']]
        enhanced_cross_inst = [r['leakage']['cross_instrument_rate'] for r in self.results['enhanced']]

        summary['leakage_reduction'] = {
            'baseline_mean': np.mean(baseline_cross_inst),
            'enhanced_mean': np.mean(enhanced_cross_inst),
            'mean_reduction': np.mean(baseline_cross_inst) - np.mean(enhanced_cross_inst),
            'median_reduction': np.median(baseline_cross_inst) - np.median(enhanced_cross_inst),
            'std_baseline': np.std(baseline_cross_inst),
            'std_enhanced': np.std(enhanced_cross_inst)
        }

        # Aggregate accuracy if available
        if 'accuracy' in self.results['baseline'][0]:
            baseline_f1 = [r['accuracy']['f1_score'] for r in self.results['baseline']]
            enhanced_f1 = [r['accuracy']['f1_score'] for r in self.results['enhanced']]

            summary['accuracy_improvement'] = {
                'baseline_f1_mean': np.mean(baseline_f1),
                'enhanced_f1_mean': np.mean(enhanced_f1),
                'f1_improvement': np.mean(enhanced_f1) - np.mean(baseline_f1),
                'improvement_percentage': ((np.mean(enhanced_f1) - np.mean(baseline_f1)) / np.mean(baseline_f1)) * 100
            }

        return summary

    def generate_visualizations(self):
        """Create comparison charts"""

        # 1. Leakage Reduction Comparison
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        baseline_leakage = [r['leakage']['cross_instrument_rate'] for r in self.results['baseline']]
        enhanced_leakage = [r['leakage']['cross_instrument_rate'] for r in self.results['enhanced']]

        # Box plot comparison
        axes[0, 0].boxplot([baseline_leakage, enhanced_leakage],
                           labels=['Baseline MR-MT3', 'Laplace Enhanced'])
        axes[0, 0].set_ylabel('Cross-Instrument Leakage Rate')
        axes[0, 0].set_title('Instrument Leakage Comparison')
        axes[0, 0].grid(True, alpha=0.3)

        # Scatter plot: baseline vs enhanced
        axes[0, 1].scatter(baseline_leakage, enhanced_leakage, alpha=0.6)
        axes[0, 1].plot([0, max(baseline_leakage)], [0, max(baseline_leakage)],
                        'r--', label='No improvement line')
        axes[0, 1].set_xlabel('Baseline Leakage Rate')
        axes[0, 1].set_ylabel('Enhanced Leakage Rate')
        axes[0, 1].set_title('Per-File Leakage Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Improvement distribution
        improvements = np.array(baseline_leakage) - np.array(enhanced_leakage)
        axes[1, 0].hist(improvements, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(np.mean(improvements), color='r',
                          linestyle='--', label=f'Mean: {np.mean(improvements):.3f}')
        axes[1, 0].set_xlabel('Leakage Reduction')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Leakage Improvements')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Cumulative improvement
        sorted_improvements = np.sort(improvements)
        cumulative = np.arange(1, len(sorted_improvements) + 1) / len(sorted_improvements)
        axes[1, 1].plot(sorted_improvements, cumulative, linewidth=2)
        axes[1, 1].axvline(0, color='r', linestyle='--', label='No improvement')
        axes[1, 1].set_xlabel('Leakage Reduction')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Cumulative Distribution of Improvements')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'leakage_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Accuracy Comparison (if available)
        if 'accuracy' in self.results['baseline'][0]:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            metrics_names = ['precision', 'recall', 'f1_score']
            for idx, metric in enumerate(metrics_names):
                baseline_vals = [r['accuracy'][metric] for r in self.results['baseline']]
                enhanced_vals = [r['accuracy'][metric] for r in self.results['enhanced']]

                axes[idx].boxplot([baseline_vals, enhanced_vals],
                                 labels=['Baseline', 'Enhanced'])
                axes[idx].set_ylabel(metric.replace('_', ' ').title())
                axes[idx].set_title(f'{metric.replace("_", " ").title()} Comparison')
                axes[idx].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

    def generate_html_report(self, summary: Dict):
        """Generate HTML report with all results"""

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>MR-MT3 + Laplace Enhancement Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   color: white; padding: 30px; border-radius: 10px; }}
        .section {{ background: white; margin: 20px 0; padding: 20px;
                    border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric {{ display: inline-block; margin: 10px 20px; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .metric-label {{ color: #666; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #667eea; color: white; }}
        .improvement-positive {{ color: #10b981; font-weight: bold; }}
        .improvement-negative {{ color: #ef4444; font-weight: bold; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>MR-MT3 + Laplace Enhancement Evaluation Report</h1>
        <p>Comparative analysis of baseline MR-MT3 vs Laplace-enhanced transcriptions</p>
    </div>

    <div class="section">
        <h2>📊 Summary Statistics</h2>
        <div class="metric">
            <div class="metric-value">{self.results['metadata']['total_files']}</div>
            <div class="metric-label">Total Files Evaluated</div>
        </div>
        <div class="metric">
            <div class="metric-value">{self.results['metadata']['successful_comparisons']}</div>
            <div class="metric-label">Successful Comparisons</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary['leakage_reduction']['mean_reduction']:.3f}</div>
            <div class="metric-label">Mean Leakage Reduction</div>
        </div>
    </div>

    <div class="section">
        <h2>🎯 Instrument Leakage Reduction</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Baseline MR-MT3</th>
                <th>Laplace Enhanced</th>
                <th>Improvement</th>
            </tr>
            <tr>
                <td>Mean Cross-Instrument Rate</td>
                <td>{summary['leakage_reduction']['baseline_mean']:.4f}</td>
                <td>{summary['leakage_reduction']['enhanced_mean']:.4f}</td>
                <td class="improvement-positive">{summary['leakage_reduction']['mean_reduction']:.4f}</td>
            </tr>
            <tr>
                <td>Median Reduction</td>
                <td colspan="2">-</td>
                <td class="improvement-positive">{summary['leakage_reduction']['median_reduction']:.4f}</td>
            </tr>
            <tr>
                <td>Standard Deviation</td>
                <td>{summary['leakage_reduction']['std_baseline']:.4f}</td>
                <td>{summary['leakage_reduction']['std_enhanced']:.4f}</td>
                <td>-</td>
            </tr>
        </table>
        <img src="leakage_comparison.png" alt="Leakage Comparison Charts">
    </div>
"""

        # Add accuracy section if available
        if summary.get('accuracy_improvement'):
            html_content += f"""
    <div class="section">
        <h2>✓ Note Accuracy Improvement</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Baseline MR-MT3</th>
                <th>Laplace Enhanced</th>
                <th>Improvement</th>
            </tr>
            <tr>
                <td>F1 Score</td>
                <td>{summary['accuracy_improvement']['baseline_f1_mean']:.4f}</td>
                <td>{summary['accuracy_improvement']['enhanced_f1_mean']:.4f}</td>
                <td class="improvement-positive">+{summary['accuracy_improvement']['f1_improvement']:.4f}
                    ({summary['accuracy_improvement']['improvement_percentage']:.2f}%)</td>
            </tr>
        </table>
        <img src="accuracy_comparison.png" alt="Accuracy Comparison Charts">
    </div>
"""

        html_content += """
    <div class="section">
        <h2>📝 Methodology</h2>
        <p><strong>Baseline:</strong> Raw MR-MT3 transcriptions without post-processing</p>
        <p><strong>Enhanced:</strong> MR-MT3 transcriptions with Laplace transform-based enhancement</p>
        <h3>Enhancement Pipeline Stages:</h3>
        <ol>
            <li><strong>Feature Extraction:</strong> Prony analysis (decay rates), VQT (spectral), Gammatone (attack times)</li>
            <li><strong>Consolidation:</strong> Merge instruments with similar Laplace characteristics</li>
            <li><strong>Refinement:</strong> Remove leaked notes based on temporal/spectral mismatch</li>
            <li><strong>Metrics:</strong> Comprehensive evaluation of improvements</li>
        </ol>
    </div>

    <div class="section">
        <h2>🔍 Key Findings</h2>
        <ul>
"""

        # Add key findings based on results
        if summary['leakage_reduction']['mean_reduction'] > 0:
            html_content += f"""
            <li class="improvement-positive">✓ Laplace enhancement reduced instrument leakage by
                {summary['leakage_reduction']['mean_reduction']:.3f} on average</li>
"""

        if summary.get('accuracy_improvement') and summary['accuracy_improvement']['f1_improvement'] > 0:
            html_content += f"""
            <li class="improvement-positive">✓ Note accuracy (F1) improved by
                {summary['accuracy_improvement']['improvement_percentage']:.2f}%</li>
"""

        html_content += """
        </ul>
    </div>

    <div class="section">
        <p style="color: #666; font-size: 0.9em;">
            Generated by evaluate_enhancement.py |
            MR-MT3 + Laplace Enhancement Pipeline
        </p>
    </div>
</body>
</html>
"""

        report_path = self.output_dir / 'evaluation_report.html'
        with open(report_path, 'w') as f:
            f.write(html_content)

        print(f"HTML report saved to: {report_path}")

    def save_results(self):
        """Save detailed results to JSON"""

        results_path = self.output_dir / 'detailed_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"Detailed results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate MR-MT3 + Laplace Enhancement pipeline'
    )
    parser.add_argument('--baseline-dir', type=str, required=True,
                       help='Directory with baseline MR-MT3 MIDI outputs')
    parser.add_argument('--enhanced-dir', type=str, required=True,
                       help='Directory with Laplace-enhanced MIDI outputs')
    parser.add_argument('--ground-truth-dir', type=str,
                       help='Directory with ground truth MIDI files (optional)')
    parser.add_argument('--audio-dir', type=str,
                       help='Directory with original audio files (optional)')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                       help='Output directory for evaluation results')

    args = parser.parse_args()

    # Create evaluator
    evaluator = EnhancementEvaluator(Path(args.output_dir))

    # Run evaluation
    print("Starting evaluation...")
    evaluator.evaluate_dataset(
        Path(args.baseline_dir),
        Path(args.enhanced_dir),
        Path(args.ground_truth_dir) if args.ground_truth_dir else None,
        Path(args.audio_dir) if args.audio_dir else None
    )

    # Generate summary
    print("Generating summary statistics...")
    summary = evaluator.generate_summary_statistics()

    # Create visualizations
    print("Creating visualizations...")
    evaluator.generate_visualizations()

    # Generate reports
    print("Generating HTML report...")
    evaluator.generate_html_report(summary)

    # Save detailed results
    evaluator.save_results()

    print(f"\nEvaluation complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Open {args.output_dir}/evaluation_report.html to view the report")


if __name__ == '__main__':
    main()
