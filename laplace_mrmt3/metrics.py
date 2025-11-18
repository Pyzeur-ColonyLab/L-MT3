# -*- coding: utf-8 -*-
"""
Evaluation Metrics for Phase 1 MR-MT3 Laplace Enhancement

Implements three key metrics to measure improvement over vanilla MR-MT3:
1. Instrument Leakage Ratio (phi) - measures over-prediction
2. Decay Consistency Score - measures exponential decay variance
3. Timbre Homogeneity - measures spectral variance within instruments

Statistical rigor: All metrics include confidence intervals and significance tests.
"""

import numpy as np
import pretty_midi
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import warnings

from .config import EnhancementConfig, MetricsConfig


@dataclass
class MetricResult:
    """Container for individual metric results with statistical measures"""
    value: float
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    std_error: float = 0.0
    n_samples: int = 0
    interpretation: str = ""

    def __repr__(self) -> str:
        return (f"MetricResult(value={self.value:.4f}, "
                f"CI={self.confidence_interval}, n={self.n_samples})")


@dataclass
class EnhancementReport:
    """Complete evaluation report comparing vanilla vs enhanced transcription"""
    # Leakage metrics
    leakage_ratio: MetricResult = None
    leakage_improvement_pct: float = 0.0
    leakage_meets_target: bool = False

    # Decay consistency
    decay_consistency: MetricResult = None
    decay_meets_target: bool = False

    # Timbre homogeneity
    timbre_homogeneity: MetricResult = None
    timbre_meets_target: bool = False

    # Overall assessment
    overall_improvement: float = 0.0
    phase1_success: bool = False

    # Raw data for analysis
    per_instrument_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate human-readable summary"""
        lines = [
            "=" * 70,
            "PHASE 1 ENHANCEMENT EVALUATION REPORT",
            "=" * 70,
            "",
            f"1. INSTRUMENT LEAKAGE RATIO (phi)",
            f"   Value: {self.leakage_ratio.value:.4f} (baseline: 1.24)",
            f"   95% CI: [{self.leakage_ratio.confidence_interval[0]:.4f}, "
            f"{self.leakage_ratio.confidence_interval[1]:.4f}]",
            f"   Improvement: {self.leakage_improvement_pct:.1f}%",
            f"   Target met: {'YES' if self.leakage_meets_target else 'NO (target: <1.12)'}",
            f"   Interpretation: {self.leakage_ratio.interpretation}",
            "",
            f"2. DECAY CONSISTENCY",
            f"   Value: {self.decay_consistency.value:.4f}",
            f"   95% CI: [{self.decay_consistency.confidence_interval[0]:.4f}, "
            f"{self.decay_consistency.confidence_interval[1]:.4f}]",
            f"   Target met: {'YES' if self.decay_meets_target else 'NO (target: >0.75)'}",
            f"   Interpretation: {self.decay_consistency.interpretation}",
            "",
            f"3. TIMBRE HOMOGENEITY",
            f"   Value: {self.timbre_homogeneity.value:.4f}",
            f"   95% CI: [{self.timbre_homogeneity.confidence_interval[0]:.4f}, "
            f"{self.timbre_homogeneity.confidence_interval[1]:.4f}]",
            f"   Target met: {'YES' if self.timbre_meets_target else 'NO (target: >0.75)'}",
            f"   Interpretation: {self.timbre_homogeneity.interpretation}",
            "",
            f"OVERALL PHASE 1 SUCCESS: {'PASS' if self.phase1_success else 'FAIL'}",
            f"Overall improvement score: {self.overall_improvement:.2f}",
            "=" * 70,
        ]
        return "\n".join(lines)


class EnhancementMetrics:
    """
    Comprehensive metrics computation for Phase 1 enhancement evaluation

    All metrics are computed with statistical rigor including confidence intervals
    and significance testing where applicable.
    """

    def __init__(self, config: Optional[MetricsConfig] = None):
        """
        Initialize metrics calculator

        Args:
            config: Metrics configuration (uses defaults if None)
        """
        self.config = config or MetricsConfig()

    def compute_leakage_ratio(
        self,
        predicted_instruments: List[int],
        ground_truth_instruments: List[int]
    ) -> MetricResult:
        """
        Compute Instrument Leakage Ratio (phi)

        phi = n_predicted_instruments / n_true_instruments
        - phi = 1.0: perfect prediction
        - phi > 1.0: over-prediction (leakage)
        - phi < 1.0: under-prediction (missed instruments)

        MR-MT3 baseline: phi = 1.24
        Phase 1 target: phi < 1.12 (>10% improvement)

        Args:
            predicted_instruments: List of unique MIDI program numbers from transcription
            ground_truth_instruments: List of true instrument program numbers

        Returns:
            MetricResult with leakage ratio and confidence interval
        """
        n_pred = len(set(predicted_instruments))
        n_true = len(set(ground_truth_instruments))

        if n_true == 0:
            warnings.warn("Ground truth has zero instruments, cannot compute leakage ratio")
            return MetricResult(
                value=float('inf'),
                interpretation="Invalid: no ground truth instruments"
            )

        phi = n_pred / n_true

        # Bootstrap confidence interval (1000 resamples)
        # Assuming Poisson-distributed counts for instrument detection
        bootstrap_samples = []
        rng = np.random.default_rng(42)
        for _ in range(1000):
            # Resample predicted instruments with replacement
            sample_pred = rng.choice(predicted_instruments, size=len(predicted_instruments), replace=True)
            sample_n_pred = len(set(sample_pred))
            bootstrap_samples.append(sample_n_pred / n_true)

        bootstrap_samples = np.array(bootstrap_samples)
        ci_lower = np.percentile(bootstrap_samples, 2.5)
        ci_upper = np.percentile(bootstrap_samples, 97.5)
        std_error = np.std(bootstrap_samples)

        # Interpretation
        if phi <= 1.0:
            interp = "Perfect or under-prediction (no leakage)"
        elif phi < 1.12:
            interp = f"Excellent: {((1.24 - phi) / (1.24 - 1.0) * 100):.1f}% improvement over baseline"
        elif phi < 1.18:
            interp = f"Good: {((1.24 - phi) / (1.24 - 1.0) * 100):.1f}% improvement over baseline"
        else:
            interp = f"Poor: only {((1.24 - phi) / (1.24 - 1.0) * 100):.1f}% improvement over baseline"

        return MetricResult(
            value=phi,
            confidence_interval=(ci_lower, ci_upper),
            std_error=std_error,
            n_samples=len(predicted_instruments),
            interpretation=interp
        )

    def compute_decay_consistency(
        self,
        decay_rates: np.ndarray,
        per_instrument: bool = False,
        instrument_labels: Optional[List[int]] = None
    ) -> MetricResult:
        """
        Compute Decay Consistency Score

        Measures variance of exponential decay rates within each instrument.
        Lower variance -> more consistent -> better separation.

        cv = std(decay_rates) / |mean(decay_rates)|  (coefficient of variation)
        consistency = 1 / (1 + cv)

        - consistency = 1.0: perfect (no variance)
        - consistency -> 0: high variance

        Phase 1 target: > 0.75

        Args:
            decay_rates: Array of decay rate values (positive) from Prony analysis
            per_instrument: If True, compute per-instrument consistency
            instrument_labels: MIDI program numbers for each decay rate

        Returns:
            MetricResult with consistency score and confidence interval
        """
        if len(decay_rates) == 0:
            warnings.warn("No decay rates provided")
            return MetricResult(
                value=0.0,
                interpretation="Invalid: no decay data"
            )

        # Remove invalid values
        valid_mask = np.isfinite(decay_rates) & (decay_rates > 0)
        decay_rates = decay_rates[valid_mask]

        if len(decay_rates) < 2:
            warnings.warn("Insufficient valid decay rates for consistency computation")
            return MetricResult(
                value=0.0,
                interpretation="Invalid: insufficient data"
            )

        if per_instrument and instrument_labels is not None:
            # Compute per-instrument consistency and average
            instrument_labels = np.array(instrument_labels)[valid_mask]
            unique_instruments = np.unique(instrument_labels)

            consistencies = []
            for inst in unique_instruments:
                inst_mask = instrument_labels == inst
                inst_rates = decay_rates[inst_mask]

                if len(inst_rates) >= 2:
                    mean_rate = np.mean(inst_rates)
                    std_rate = np.std(inst_rates, ddof=1)

                    if mean_rate > 1e-10:  # Avoid division by zero
                        cv = std_rate / abs(mean_rate)
                        consistency = 1.0 / (1.0 + cv)
                        consistencies.append(consistency)

            if len(consistencies) == 0:
                return MetricResult(
                    value=0.0,
                    interpretation="Invalid: no valid per-instrument data"
                )

            final_consistency = np.mean(consistencies)
            std_error = np.std(consistencies, ddof=1) / np.sqrt(len(consistencies)) if len(consistencies) > 1 else 0.0

            # Confidence interval via normal approximation
            z_critical = 1.96  # 95% CI
            ci_lower = max(0.0, final_consistency - z_critical * std_error)
            ci_upper = min(1.0, final_consistency + z_critical * std_error)

        else:
            # Global consistency across all instruments
            mean_rate = np.mean(decay_rates)
            std_rate = np.std(decay_rates, ddof=1)

            if abs(mean_rate) < 1e-10:
                return MetricResult(
                    value=0.0,
                    interpretation="Invalid: zero mean decay rate"
                )

            cv = std_rate / abs(mean_rate)
            final_consistency = 1.0 / (1.0 + cv)

            # Bootstrap confidence interval
            bootstrap_samples = []
            rng = np.random.default_rng(42)
            for _ in range(1000):
                sample = rng.choice(decay_rates, size=len(decay_rates), replace=True)
                sample_mean = np.mean(sample)
                sample_std = np.std(sample, ddof=1)
                if abs(sample_mean) > 1e-10:
                    sample_cv = sample_std / abs(sample_mean)
                    bootstrap_samples.append(1.0 / (1.0 + sample_cv))

            bootstrap_samples = np.array(bootstrap_samples)
            ci_lower = np.percentile(bootstrap_samples, 2.5)
            ci_upper = np.percentile(bootstrap_samples, 97.5)
            std_error = np.std(bootstrap_samples)

        # Interpretation
        if final_consistency >= 0.80:
            interp = "Excellent: very consistent decay patterns"
        elif final_consistency >= 0.75:
            interp = "Good: meets target consistency"
        elif final_consistency >= 0.65:
            interp = "Acceptable: near-target consistency"
        else:
            interp = "Poor: high decay variance, weak separation"

        return MetricResult(
            value=final_consistency,
            confidence_interval=(ci_lower, ci_upper),
            std_error=std_error,
            n_samples=len(decay_rates),
            interpretation=interp
        )

    def compute_timbre_homogeneity(
        self,
        spectral_centroids: np.ndarray,
        per_instrument: bool = False,
        instrument_labels: Optional[List[int]] = None
    ) -> MetricResult:
        """
        Compute Timbre Homogeneity Score

        Measures spectral variance within each instrument track.
        Lower variance -> more homogeneous -> better separation.

        homogeneity = 1 - (std(centroids) / mean(centroids))

        - homogeneity = 1.0: perfect (no variance)
        - homogeneity -> 0: high variance

        Phase 1 target: > 0.75

        Args:
            spectral_centroids: Array of spectral centroid values (Hz)
            per_instrument: If True, compute per-instrument homogeneity
            instrument_labels: MIDI program numbers for each centroid

        Returns:
            MetricResult with homogeneity score and confidence interval
        """
        if len(spectral_centroids) == 0:
            warnings.warn("No spectral centroids provided")
            return MetricResult(
                value=0.0,
                interpretation="Invalid: no spectral data"
            )

        # Remove invalid values
        valid_mask = np.isfinite(spectral_centroids) & (spectral_centroids > 0)
        spectral_centroids = spectral_centroids[valid_mask]

        if len(spectral_centroids) < 2:
            warnings.warn("Insufficient valid centroids for homogeneity computation")
            return MetricResult(
                value=0.0,
                interpretation="Invalid: insufficient data"
            )

        if per_instrument and instrument_labels is not None:
            # Compute per-instrument homogeneity and average
            instrument_labels = np.array(instrument_labels)[valid_mask]
            unique_instruments = np.unique(instrument_labels)

            homogeneities = []
            for inst in unique_instruments:
                inst_mask = instrument_labels == inst
                inst_centroids = spectral_centroids[inst_mask]

                if len(inst_centroids) >= 2:
                    mean_centroid = np.mean(inst_centroids)
                    std_centroid = np.std(inst_centroids, ddof=1)

                    if mean_centroid > 1e-10:  # Avoid division by zero
                        homogeneity = 1.0 - (std_centroid / mean_centroid)
                        # Clamp to [0, 1] (variance can exceed mean)
                        homogeneity = max(0.0, min(1.0, homogeneity))
                        homogeneities.append(homogeneity)

            if len(homogeneities) == 0:
                return MetricResult(
                    value=0.0,
                    interpretation="Invalid: no valid per-instrument data"
                )

            final_homogeneity = np.mean(homogeneities)
            std_error = np.std(homogeneities, ddof=1) / np.sqrt(len(homogeneities)) if len(homogeneities) > 1 else 0.0

            # Confidence interval via normal approximation
            z_critical = 1.96  # 95% CI
            ci_lower = max(0.0, final_homogeneity - z_critical * std_error)
            ci_upper = min(1.0, final_homogeneity + z_critical * std_error)

        else:
            # Global homogeneity across all instruments
            mean_centroid = np.mean(spectral_centroids)
            std_centroid = np.std(spectral_centroids, ddof=1)

            if mean_centroid < 1e-10:
                return MetricResult(
                    value=0.0,
                    interpretation="Invalid: zero mean spectral centroid"
                )

            final_homogeneity = 1.0 - (std_centroid / mean_centroid)
            final_homogeneity = max(0.0, min(1.0, final_homogeneity))

            # Bootstrap confidence interval
            bootstrap_samples = []
            rng = np.random.default_rng(42)
            for _ in range(1000):
                sample = rng.choice(spectral_centroids, size=len(spectral_centroids), replace=True)
                sample_mean = np.mean(sample)
                sample_std = np.std(sample, ddof=1)
                if sample_mean > 1e-10:
                    homog = 1.0 - (sample_std / sample_mean)
                    bootstrap_samples.append(max(0.0, min(1.0, homog)))

            bootstrap_samples = np.array(bootstrap_samples)
            ci_lower = np.percentile(bootstrap_samples, 2.5)
            ci_upper = np.percentile(bootstrap_samples, 97.5)
            std_error = np.std(bootstrap_samples)

        # Interpretation
        if final_homogeneity >= 0.80:
            interp = "Excellent: very homogeneous timbre within instruments"
        elif final_homogeneity >= 0.75:
            interp = "Good: meets target homogeneity"
        elif final_homogeneity >= 0.65:
            interp = "Acceptable: moderate spectral consistency"
        else:
            interp = "Poor: high spectral variance, weak separation"

        return MetricResult(
            value=final_homogeneity,
            confidence_interval=(ci_lower, ci_upper),
            std_error=std_error,
            n_samples=len(spectral_centroids),
            interpretation=interp
        )

    def evaluate_enhancement(
        self,
        predicted_instruments: List[int],
        ground_truth_instruments: List[int],
        decay_rates: np.ndarray,
        spectral_centroids: np.ndarray,
        instrument_labels_decay: Optional[List[int]] = None,
        instrument_labels_spectral: Optional[List[int]] = None,
    ) -> EnhancementReport:
        """
        Compute all Phase 1 metrics and generate evaluation report

        Args:
            predicted_instruments: Predicted MIDI program numbers
            ground_truth_instruments: True MIDI program numbers
            decay_rates: Array of decay rates from Prony analysis
            spectral_centroids: Array of spectral centroids (Hz)
            instrument_labels_decay: Per-note instrument labels for decay rates
            instrument_labels_spectral: Per-note instrument labels for centroids

        Returns:
            EnhancementReport with all metrics and overall assessment
        """
        report = EnhancementReport()

        # 1. Leakage ratio
        report.leakage_ratio = self.compute_leakage_ratio(
            predicted_instruments,
            ground_truth_instruments
        )

        # Compute improvement percentage
        phi = report.leakage_ratio.value
        baseline = self.config.phi_baseline
        improvement = ((baseline - phi) / (baseline - 1.0)) * 100  # % of gap closed
        report.leakage_improvement_pct = improvement
        report.leakage_meets_target = phi < self.config.phi_target_goal

        # 2. Decay consistency
        report.decay_consistency = self.compute_decay_consistency(
            decay_rates,
            per_instrument=True if instrument_labels_decay is not None else False,
            instrument_labels=instrument_labels_decay
        )
        report.decay_meets_target = report.decay_consistency.value >= self.config.decay_consistency_goal

        # 3. Timbre homogeneity
        report.timbre_homogeneity = self.compute_timbre_homogeneity(
            spectral_centroids,
            per_instrument=True if instrument_labels_spectral is not None else False,
            instrument_labels=instrument_labels_spectral
        )
        report.timbre_meets_target = report.timbre_homogeneity.value >= self.config.timbre_homogeneity_goal

        # Overall assessment
        targets_met = sum([
            report.leakage_meets_target,
            report.decay_meets_target,
            report.timbre_meets_target
        ])

        # Weighted overall score (leakage is most important)
        leakage_score = max(0, (baseline - phi) / (baseline - 1.0))  # 0-1 scale
        decay_score = report.decay_consistency.value
        timbre_score = report.timbre_homogeneity.value

        report.overall_improvement = (
            0.5 * leakage_score +  # 50% weight on leakage
            0.25 * decay_score +   # 25% weight on consistency
            0.25 * timbre_score    # 25% weight on homogeneity
        )

        # Phase 1 success requires meeting at least 2/3 targets with leakage being one
        report.phase1_success = (
            report.leakage_meets_target and
            targets_met >= 2
        )

        return report

    def print_report(self, report: EnhancementReport):
        """Pretty-print evaluation report"""
        print(report.summary())


def compare_transcriptions(
    vanilla_midi_path: str,
    enhanced_midi_path: str,
    ground_truth_midi_path: str,
    decay_rates: np.ndarray,
    spectral_centroids: np.ndarray,
    config: Optional[MetricsConfig] = None,
    instrument_labels_decay: Optional[List[int]] = None,
    instrument_labels_spectral: Optional[List[int]] = None,
) -> EnhancementReport:
    """
    Compare vanilla vs enhanced transcription against ground truth

    Args:
        vanilla_midi_path: Path to vanilla MR-MT3 transcription
        enhanced_midi_path: Path to Laplace-enhanced transcription
        ground_truth_midi_path: Path to ground truth MIDI
        decay_rates: Decay rates from Prony analysis
        spectral_centroids: Spectral centroids from feature extraction
        config: Metrics configuration
        instrument_labels_decay: Per-note instrument labels for decay
        instrument_labels_spectral: Per-note instrument labels for spectral

    Returns:
        EnhancementReport with comparative analysis
    """
    # Load MIDI files
    vanilla_midi = pretty_midi.PrettyMIDI(vanilla_midi_path)
    enhanced_midi = pretty_midi.PrettyMIDI(enhanced_midi_path)
    ground_truth_midi = pretty_midi.PrettyMIDI(ground_truth_midi_path)

    # Extract instrument programs
    vanilla_instruments = [inst.program for inst in vanilla_midi.instruments if not inst.is_drum]
    enhanced_instruments = [inst.program for inst in enhanced_midi.instruments if not inst.is_drum]
    ground_truth_instruments = [inst.program for inst in ground_truth_midi.instruments if not inst.is_drum]

    # Compute metrics for enhanced version
    metrics_calc = EnhancementMetrics(config)
    report = metrics_calc.evaluate_enhancement(
        predicted_instruments=enhanced_instruments,
        ground_truth_instruments=ground_truth_instruments,
        decay_rates=decay_rates,
        spectral_centroids=spectral_centroids,
        instrument_labels_decay=instrument_labels_decay,
        instrument_labels_spectral=instrument_labels_spectral
    )

    # Add vanilla comparison
    vanilla_leakage = metrics_calc.compute_leakage_ratio(
        vanilla_instruments,
        ground_truth_instruments
    )

    print("\n" + "=" * 70)
    print("VANILLA vs ENHANCED COMPARISON")
    print("=" * 70)
    print(f"Vanilla leakage ratio: {vanilla_leakage.value:.4f}")
    print(f"Enhanced leakage ratio: {report.leakage_ratio.value:.4f}")
    print(f"Absolute improvement: {vanilla_leakage.value - report.leakage_ratio.value:.4f}")
    print(f"Relative improvement: {((vanilla_leakage.value - report.leakage_ratio.value) / vanilla_leakage.value * 100):.1f}%")
    print("=" * 70)

    return report


def compute_per_instrument_metrics(
    midi_path: str,
    decay_rates_per_note: Dict[int, List[float]],
    spectral_centroids_per_note: Dict[int, List[float]],
    config: Optional[MetricsConfig] = None
) -> Dict[int, Dict[str, MetricResult]]:
    """
    Compute metrics for each instrument separately

    Args:
        midi_path: Path to MIDI transcription
        decay_rates_per_note: Dict mapping instrument program to list of decay rates
        spectral_centroids_per_note: Dict mapping instrument program to list of centroids
        config: Metrics configuration

    Returns:
        Dict mapping instrument program to metrics dict
    """
    metrics_calc = EnhancementMetrics(config)
    per_instrument = {}

    midi = pretty_midi.PrettyMIDI(midi_path)

    for inst in midi.instruments:
        if inst.is_drum:
            continue

        program = inst.program

        # Get decay rates for this instrument
        decay_rates = np.array(decay_rates_per_note.get(program, []))
        spectral_centroids = np.array(spectral_centroids_per_note.get(program, []))

        inst_metrics = {}

        if len(decay_rates) > 0:
            inst_metrics['decay_consistency'] = metrics_calc.compute_decay_consistency(decay_rates)

        if len(spectral_centroids) > 0:
            inst_metrics['timbre_homogeneity'] = metrics_calc.compute_timbre_homogeneity(spectral_centroids)

        per_instrument[program] = inst_metrics

    return per_instrument


if __name__ == "__main__":
    """Demo and validation of metrics computation"""

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
