# -*- coding: utf-8 -*-
"""
Configuration management for Phase 1 MR-MT3 Laplace Enhancement

Centralizes all tunable parameters for consolidation, refinement, and feature extraction.
Based on PHASE1_MRMT3_SPECIFICATION.md Appendix A: Parameter Reference
"""

import yaml
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class PronyConfig:
    """Prony Analysis parameters"""
    n_components: int = 10  # Number of exponential components to extract [5-20]
    model_order: int = 30  # Linear prediction model order [20-50]
    min_note_duration: float = 0.05  # Minimum note duration in seconds
    hop_length: int = 256  # Hop length for STFT-style analysis
    win_length: int = 1024  # Window length for analysis


@dataclass
class VQTConfig:
    """Variable-Q Transform parameters"""
    fmin: float = 55.0  # Minimum frequency in Hz (A1)
    n_bins: int = 72  # Number of frequency bins (6 octaves)
    bins_per_octave: int = 12  # Semitones per octave
    q_rate: float = 1.0  # Q factor for frequency/time resolution trade-off


@dataclass
class GammatoneConfig:
    """Gammatone Filterbank parameters"""
    n_filters: int = 48  # Number of filters [32-64]
    fmin: float = 50.0  # Minimum center frequency (Hz)
    fmax: float = 4000.0  # Maximum center frequency (Hz)


@dataclass
class ConsolidationConfig:
    """Decay-based consolidation parameters"""
    # Similarity thresholds
    decay_similarity_threshold: float = 0.80  # [0.7-0.9] - within 20% decay difference
    spectral_diff_threshold: float = 200.0  # Hz [150-300] - spectral centroid similarity
    midi_class_match_required: bool = True  # Require same MIDI instrument family

    # Strategy selection
    strategy: str = "conservative"  # "conservative", "balanced", "aggressive"

    # Strategy-specific overrides
    conservative_decay_threshold: float = 0.90
    conservative_spectral_threshold: float = 150.0

    balanced_decay_threshold: float = 0.80
    balanced_spectral_threshold: float = 200.0

    aggressive_decay_threshold: float = 0.70
    aggressive_spectral_threshold: float = 300.0

    # Feature fallback
    use_prony_primary: bool = True  # Try Prony first, fallback to VQT+Gammatone
    prony_failure_threshold: float = 0.20  # Max acceptable Prony failure rate

    def get_active_thresholds(self) -> Dict[str, float]:
        """Get thresholds based on selected strategy"""
        if self.strategy == "conservative":
            return {
                "decay": self.conservative_decay_threshold,
                "spectral": self.conservative_spectral_threshold
            }
        elif self.strategy == "aggressive":
            return {
                "decay": self.aggressive_decay_threshold,
                "spectral": self.aggressive_spectral_threshold
            }
        else:  # balanced (default)
            return {
                "decay": self.balanced_decay_threshold,
                "spectral": self.balanced_spectral_threshold
            }


@dataclass
class RefinementConfig:
    """Timbre-based refinement parameters"""
    # Heuristic thresholds
    dark_timbre_threshold: float = 300.0  # Hz - below this is "dark"
    mid_timbre_threshold: float = 800.0  # Hz - mid-range cutoff
    bright_timbre_threshold: float = 1500.0  # Hz - above this is "bright"

    harmonic_ratio_threshold: float = 2.0  # [1.5-3.0] - harmonic vs inharmonic
    fast_attack_threshold: float = 0.010  # seconds - percussive vs sustained

    # Refinement strategy
    enable_heuristics: bool = True  # Use rule-based refinement
    enable_classifier: bool = False  # Use ML classifier (Phase 2)

    # Phase 2: ML Classifier parameters
    min_confidence: float = 0.7  # Minimum confidence threshold for ML predictions [0.5-0.95]

    # MIDI program mappings
    bass_programs: list = field(default_factory=lambda: [32, 33, 34, 35, 36, 37, 38, 39])
    string_programs: list = field(default_factory=lambda: [40, 41, 42, 43, 44, 45, 46, 47])
    brass_programs: list = field(default_factory=lambda: [56, 57, 58, 59, 60, 61, 62, 63])
    piano_programs: list = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7])
    guitar_programs: list = field(default_factory=lambda: [24, 25, 26, 27, 28, 29, 30, 31])


@dataclass
class MetricsConfig:
    """Evaluation metrics parameters"""
    # Leakage ratio targets (�)
    phi_baseline: float = 1.24  # MR-MT3 baseline from paper
    phi_target_minimum: float = 1.18  # >5% improvement
    phi_target_goal: float = 1.12  # >10% improvement
    phi_target_excellent: float = 1.05  # >15% improvement

    # Decay consistency targets
    decay_consistency_minimum: float = 0.65
    decay_consistency_goal: float = 0.75
    decay_consistency_excellent: float = 0.80

    # Timbre homogeneity targets
    timbre_homogeneity_baseline: float = 0.55  # Estimated vanilla MR-MT3
    timbre_homogeneity_goal: float = 0.75


@dataclass
class EnhancementConfig:
    """Top-level enhancement pipeline configuration"""
    # Sub-configs
    prony: PronyConfig = field(default_factory=PronyConfig)
    vqt: VQTConfig = field(default_factory=VQTConfig)
    gammatone: GammatoneConfig = field(default_factory=GammatoneConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    # Pipeline control
    enable_consolidation: bool = True
    enable_refinement: bool = True
    enable_metrics: bool = True

    # Audio processing
    sample_rate: int = 16000  # MR-MT3 uses 16kHz

    # Logging
    verbose: bool = True
    debug: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for YAML serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnhancementConfig':
        """Load configuration from dictionary"""
        # Create sub-configs
        prony = PronyConfig(**config_dict.get('prony', {}))
        vqt = VQTConfig(**config_dict.get('vqt', {}))
        gammatone = GammatoneConfig(**config_dict.get('gammatone', {}))
        consolidation = ConsolidationConfig(**config_dict.get('consolidation', {}))
        refinement = RefinementConfig(**config_dict.get('refinement', {}))
        metrics = MetricsConfig(**config_dict.get('metrics', {}))

        # Create main config
        return cls(
            prony=prony,
            vqt=vqt,
            gammatone=gammatone,
            consolidation=consolidation,
            refinement=refinement,
            metrics=metrics,
            enable_consolidation=config_dict.get('enable_consolidation', True),
            enable_refinement=config_dict.get('enable_refinement', True),
            enable_metrics=config_dict.get('enable_metrics', True),
            sample_rate=config_dict.get('sample_rate', 16000),
            verbose=config_dict.get('verbose', True),
            debug=config_dict.get('debug', False)
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'EnhancementConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

    def validate(self) -> bool:
        """Validate configuration parameters"""
        issues = []

        # Prony validation
        if not (5 <= self.prony.n_components <= 20):
            issues.append(f"Prony n_components={self.prony.n_components} outside recommended range [5-20]")
        if not (20 <= self.prony.model_order <= 50):
            issues.append(f"Prony model_order={self.prony.model_order} outside recommended range [20-50]")

        # Consolidation validation
        if not (0.7 <= self.consolidation.decay_similarity_threshold <= 0.9):
            issues.append(f"Decay threshold={self.consolidation.decay_similarity_threshold} outside range [0.7-0.9]")
        if not (150 <= self.consolidation.spectral_diff_threshold <= 300):
            issues.append(f"Spectral threshold={self.consolidation.spectral_diff_threshold} outside range [150-300]")

        # Refinement validation
        if not (1.5 <= self.refinement.harmonic_ratio_threshold <= 3.0):
            issues.append(f"Harmonic ratio={self.refinement.harmonic_ratio_threshold} outside range [1.5-3.0]")

        if issues:
            if self.verbose:
                print("�  Configuration validation warnings:")
                for issue in issues:
                    print(f"  - {issue}")
            return False

        return True


def create_default_config_file(output_path: str = "configs/enhancement.yaml"):
    """Create a default configuration file with documentation"""
    config = EnhancementConfig()

    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    config.to_yaml(output_path)
    print(f" Default configuration created: {output_path}")
    return config


if __name__ == "__main__":
    # Demo: Create default configuration
    config = create_default_config_file("../configs/enhancement.yaml")

    # Validate
    is_valid = config.validate()
    print(f"\n Configuration valid: {is_valid}")

    # Show active thresholds
    thresholds = config.consolidation.get_active_thresholds()
    print(f"\nActive consolidation thresholds (strategy={config.consolidation.strategy}):")
    print(f"  - Decay similarity: {thresholds['decay']}")
    print(f"  - Spectral difference: {thresholds['spectral']} Hz")
