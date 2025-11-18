# -*- coding: utf-8 -*-
"""
Timbre-Based Refinement System for Phase 1 MR-MT3 Enhancement

Validates and corrects instrument assignments using heuristic rules based on
extracted acoustic features (spectral centroid, attack time, harmonic ratio).

Implements Section 5 of PHASE1_MRMT3_SPECIFICATION.md:
- Rule 1: Dark, Harmonic -> Bass
- Rule 2: Mid-range, Strong Harmonics -> Strings
- Rule 3: Bright, Rich Spectrum -> Brass
- Rule 4: Fast Attack, Percussive -> Guitar/Piano

Design principles:
- Conservative: Only refine when confidence is high
- Transparent: Log all refinement decisions
- Configurable: All thresholds tunable via RefinementConfig
"""

import numpy as np
import pretty_midi
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

from .config import EnhancementConfig, RefinementConfig
from .features import MIDIAwareFeatureExtractor


@dataclass
class RefinementDecision:
    """Record of a single refinement decision"""
    instrument_idx: int
    original_program: int
    refined_program: int
    rule_applied: str
    confidence: float
    features: Dict[str, float]


class TimbreRefiner:
    """
    Apply timbre-based heuristic rules to refine MIDI instrument assignments

    Uses extracted features (centroid, attack, harmonic ratio) to correct
    instrument programs that don't match the acoustic characteristics of
    the audio source.
    """

    def __init__(self, config: EnhancementConfig):
        """
        Initialize refiner with configuration

        Args:
            config: EnhancementConfig with refinement parameters
        """
        self.config = config
        self.refinement_config = config.refinement
        self.feature_extractor = MIDIAwareFeatureExtractor(config)

        # Track refinement decisions for logging
        self.decisions: List[RefinementDecision] = []

    def classify_instrument(
        self,
        mean_centroid: float,
        mean_attack: float,
        harmonic_ratio: float
    ) -> Tuple[int, str, float]:
        """
        Classify instrument based on acoustic features using heuristic rules

        Args:
            mean_centroid: Average spectral centroid (Hz)
            mean_attack: Average attack time (seconds)
            harmonic_ratio: Harmonic-to-noise ratio

        Returns:
            Tuple of (program_number, rule_name, confidence_score)

        Rules (in priority order):
        1. Dark, Harmonic -> Bass (centroid < 300 Hz, harmonic_ratio > 2.0)
        2. Mid-range, Strong Harmonics -> Strings (300 < centroid < 800, harmonic_ratio > 3.0)
        3. Bright, Rich Spectrum -> Brass (centroid > 1500, harmonic_ratio > 2.5)
        4. Fast Attack + Bright -> Guitar (attack < 10ms, centroid > 1000)
        5. Fast Attack + Dark -> Piano (attack < 10ms, centroid <= 1000)
        """
        cfg = self.refinement_config

        # Rule 1: Dark, Harmonic -> Bass
        if (mean_centroid < cfg.dark_timbre_threshold and
            harmonic_ratio > cfg.harmonic_ratio_threshold):
            confidence = min(
                (cfg.dark_timbre_threshold - mean_centroid) / cfg.dark_timbre_threshold,
                (harmonic_ratio - cfg.harmonic_ratio_threshold) / cfg.harmonic_ratio_threshold
            )
            return (32, "dark_harmonic_bass", float(confidence))

        # Rule 2: Mid-range, Strong Harmonics -> Strings
        if (cfg.dark_timbre_threshold <= mean_centroid <= cfg.mid_timbre_threshold and
            harmonic_ratio > 3.0):  # Higher threshold for strings
            confidence = (harmonic_ratio - 3.0) / 3.0
            return (40, "midrange_harmonic_strings", float(confidence))

        # Rule 3: Bright, Rich Spectrum -> Brass
        if (mean_centroid > cfg.bright_timbre_threshold and
            harmonic_ratio > 2.5):  # Brass-specific threshold
            confidence = min(
                (mean_centroid - cfg.bright_timbre_threshold) / cfg.bright_timbre_threshold,
                (harmonic_ratio - 2.5) / 2.5
            )
            return (56, "bright_rich_brass", float(confidence))

        # Rule 4a: Fast Attack + Bright -> Guitar
        if (mean_attack < cfg.fast_attack_threshold and
            mean_centroid > 1000.0):  # Guitar threshold
            confidence = min(
                (cfg.fast_attack_threshold - mean_attack) / cfg.fast_attack_threshold,
                (mean_centroid - 1000.0) / 1000.0
            )
            return (24, "fast_bright_guitar", float(confidence))

        # Rule 4b: Fast Attack + Dark -> Piano
        if (mean_attack < cfg.fast_attack_threshold and
            mean_centroid <= 1000.0):
            confidence = (cfg.fast_attack_threshold - mean_attack) / cfg.fast_attack_threshold
            return (0, "fast_dark_piano", float(confidence))

        # No rule matches: return original program with low confidence
        return (-1, "no_rule_match", 0.0)

    def apply_heuristic_rules(
        self,
        instrument: pretty_midi.Instrument,
        inst_idx: int,
        features: Dict[str, any],
        audio: np.ndarray
    ) -> Optional[int]:
        """
        Apply heuristic rules to determine if instrument should be refined

        Args:
            instrument: pretty_midi.Instrument object
            inst_idx: Instrument index in MIDI file
            features: Extracted features dict from MIDIAwareFeatureExtractor
            audio: Raw audio for additional analysis if needed

        Returns:
            New program number if refinement is needed, None otherwise
        """
        # Skip drums
        if instrument.is_drum:
            return None

        # Validate required features are present
        if (features['mean_centroid'] is None or
            features['mean_attack'] is None):
            if self.config.verbose:
                print(f"  Warning: Skipping instrument {inst_idx}: insufficient features")
            return None

        # Compute harmonic ratio if needed
        # We'll compute it from the first note as a representative sample
        if len(instrument.notes) == 0:
            return None

        # Extract first note for harmonic analysis
        first_note = instrument.notes[0]
        start_sample = int(first_note.start * self.config.sample_rate)
        end_sample = int(first_note.end * self.config.sample_rate)

        if end_sample > len(audio):
            end_sample = len(audio)
        if start_sample >= end_sample:
            return None

        note_audio = audio[start_sample:end_sample]
        harmonic_ratio = self.feature_extractor.compute_harmonic_ratio(note_audio)

        # Classify based on features
        new_program, rule_name, confidence = self.classify_instrument(
            mean_centroid=features['mean_centroid'],
            mean_attack=features['mean_attack'],
            harmonic_ratio=harmonic_ratio
        )

        # No rule matched
        if new_program == -1:
            return None

        # Only apply refinement if confidence threshold met
        MIN_CONFIDENCE = 0.3  # Conservative threshold
        if confidence < MIN_CONFIDENCE:
            if self.config.debug:
                print(f"  Low confidence ({confidence:.2f}) for rule {rule_name}, skipping")
            return None

        # Don't change if already correct family
        original_program = instrument.program
        if self._same_instrument_family(original_program, new_program):
            return None

        # Record decision
        decision = RefinementDecision(
            instrument_idx=inst_idx,
            original_program=original_program,
            refined_program=new_program,
            rule_applied=rule_name,
            confidence=confidence,
            features={
                'centroid': features['mean_centroid'],
                'attack': features['mean_attack'],
                'harmonic_ratio': harmonic_ratio
            }
        )
        self.decisions.append(decision)

        if self.config.verbose:
            print(f"  Refinement: program {original_program} -> {new_program}")
            print(f"    Rule: {rule_name} (confidence={confidence:.2f})")
            print(f"    Features: centroid={features['mean_centroid']:.1f}Hz, "
                  f"attack={features['mean_attack']*1000:.1f}ms, "
                  f"harmonic={harmonic_ratio:.2f}")

        return new_program

    def _same_instrument_family(self, program1: int, program2: int) -> bool:
        """
        Check if two programs belong to the same General MIDI family

        General MIDI program families (each 8 programs):
        0-7: Piano, 8-15: Chromatic Percussion, 16-23: Organ, 24-31: Guitar,
        32-39: Bass, 40-47: Strings, 48-55: Ensemble, 56-63: Brass,
        64-71: Reed, 72-79: Pipe, 80-87: Synth Lead, 88-95: Synth Pad,
        96-103: Synth Effects, 104-111: Ethnic, 112-119: Percussive, 120-127: Sound Effects
        """
        return (program1 // 8) == (program2 // 8)

    def refine(
        self,
        midi: pretty_midi.PrettyMIDI,
        audio: np.ndarray,
        features: Optional[Dict[int, Dict[str, any]]] = None
    ) -> pretty_midi.PrettyMIDI:
        """
        Refine all instruments in a MIDI file based on acoustic features

        Args:
            midi: pretty_midi.PrettyMIDI object to refine
            audio: Raw audio array at config.sample_rate
            features: Pre-computed features dict (if None, will extract)

        Returns:
            New pretty_midi.PrettyMIDI object with refined instrument assignments
        """
        if not self.refinement_config.enable_heuristics:
            if self.config.verbose:
                print("Warning: Refinement disabled in config")
            return midi

        # Extract features if not provided
        if features is None:
            if self.config.verbose:
                print("\nExtracting features for refinement...")
            features = self.feature_extractor.extract_all_features(midi, audio)

        # Create new MIDI object (non-destructive)
        refined_midi = pretty_midi.PrettyMIDI(initial_tempo=midi.estimate_tempo())

        # Copy time signature and key signature
        refined_midi.time_signature_changes = midi.time_signature_changes.copy()
        refined_midi.key_signature_changes = midi.key_signature_changes.copy()

        # Process each instrument
        if self.config.verbose:
            print(f"\nApplying refinement rules to {len(midi.instruments)} instruments...")

        refinement_count = 0

        for inst_idx, instrument in enumerate(midi.instruments):
            # Create new instrument
            new_instrument = pretty_midi.Instrument(
                program=instrument.program,
                is_drum=instrument.is_drum,
                name=instrument.name
            )

            # Copy all notes
            new_instrument.notes = instrument.notes.copy()
            new_instrument.pitch_bends = instrument.pitch_bends.copy()
            new_instrument.control_changes = instrument.control_changes.copy()

            # Apply refinement
            inst_features = features.get(inst_idx, {})
            new_program = self.apply_heuristic_rules(
                instrument=instrument,
                inst_idx=inst_idx,
                features=inst_features,
                audio=audio
            )

            # Update program if refined
            if new_program is not None:
                new_instrument.program = new_program
                refinement_count += 1

            refined_midi.instruments.append(new_instrument)

        if self.config.verbose:
            print(f"\nRefinement complete: {refinement_count}/{len(midi.instruments)} instruments modified")

        return refined_midi

    def get_refinement_report(self) -> str:
        """
        Generate human-readable report of refinement decisions

        Returns:
            Formatted string with refinement statistics
        """
        if len(self.decisions) == 0:
            return "No refinement decisions made"

        report = []
        report.append("=" * 70)
        report.append("REFINEMENT REPORT")
        report.append("=" * 70)
        report.append(f"\nTotal refinements: {len(self.decisions)}")

        # Rule distribution
        rule_counts = {}
        for decision in self.decisions:
            rule_counts[decision.rule_applied] = rule_counts.get(decision.rule_applied, 0) + 1

        report.append("\nRules applied:")
        for rule, count in sorted(rule_counts.items(), key=lambda x: -x[1]):
            report.append(f"  - {rule}: {count}")

        # Average confidence
        avg_confidence = np.mean([d.confidence for d in self.decisions])
        report.append(f"\nAverage confidence: {avg_confidence:.2f}")

        # Detailed decisions
        report.append("\nDetailed decisions:")
        for decision in self.decisions:
            report.append(f"\nInstrument {decision.instrument_idx}:")
            report.append(f"  Program: {decision.original_program} -> {decision.refined_program}")
            report.append(f"  Rule: {decision.rule_applied}")
            report.append(f"  Confidence: {decision.confidence:.2f}")
            report.append(f"  Centroid: {decision.features['centroid']:.1f} Hz")
            report.append(f"  Attack: {decision.features['attack']*1000:.1f} ms")
            report.append(f"  Harmonic ratio: {decision.features['harmonic_ratio']:.2f}")

        return "\n".join(report)


def refine_by_timbre(
    midi: pretty_midi.PrettyMIDI,
    audio: np.ndarray,
    sr: int = 16000,
    config: Optional[EnhancementConfig] = None,
    features: Optional[Dict[int, Dict[str, any]]] = None
) -> Tuple[pretty_midi.PrettyMIDI, str]:
    """
    Convenience function to refine MIDI instrument assignments by timbre

    Args:
        midi: pretty_midi.PrettyMIDI object to refine
        audio: Raw audio array
        sr: Sample rate (default 16000 for MR-MT3)
        config: Enhancement configuration (creates default if None)
        features: Pre-computed features dict (if None, will extract)

    Returns:
        Tuple of (refined_midi, refinement_report_string)

    Example:
        >>> refined_midi, report = refine_by_timbre(midi, audio)
        >>> print(report)
        >>> refined_midi.write('refined_output.mid')
    """
    if config is None:
        config = EnhancementConfig()
        config.sample_rate = sr

    refiner = TimbreRefiner(config)
    refined_midi = refiner.refine(midi, audio, features)
    report = refiner.get_refinement_report()

    return refined_midi, report


if __name__ == "__main__":
    # Demo: Load babySlakh track and apply refinement
    import librosa
    from pathlib import Path
    from .features import extract_features_from_midi

    print("=" * 70)
    print("TIMBRE-BASED REFINEMENT DEMO")
    print("=" * 70)

    # Load sample from babySlakh
    track_path = Path("../babyslakh_16k/Track00001")

    # Find mix audio and MIDI files
    mix_file = list(track_path.glob("mix.wav"))
    midi_files = list((track_path / "MIDI").glob("*.mid"))

    if mix_file and midi_files:
        print(f"\nLoading: {track_path.name}")

        # Load audio
        audio, sr = librosa.load(str(mix_file[0]), sr=16000, duration=10.0)
        print(f"Audio: {len(audio)/sr:.1f}s at {sr}Hz")

        # Load first MIDI file as example
        midi = pretty_midi.PrettyMIDI(str(midi_files[0]))
        print(f"MIDI: {len(midi.instruments)} instruments")

        # Show original programs
        print("\nOriginal instrument programs:")
        for i, inst in enumerate(midi.instruments):
            print(f"  {i}: program={inst.program} ({inst.name or 'Unknown'})")

        # Extract features
        print("\nExtracting features...")
        config = EnhancementConfig(verbose=False)
        features = extract_features_from_midi(midi, audio, sr, config)

        # Apply refinement
        print("\nApplying refinement...")
        config.verbose = True
        refined_midi, report = refine_by_timbre(midi, audio, sr, config, features)

        # Show results
        print("\n" + report)

        # Show refined programs
        print("\nRefined instrument programs:")
        for i, inst in enumerate(refined_midi.instruments):
            original_program = midi.instruments[i].program
            if inst.program != original_program:
                print(f"  {i}: program={inst.program} (was {original_program}) <- REFINED")
            else:
                print(f"  {i}: program={inst.program} (unchanged)")

        # Save refined MIDI
        output_path = "refined_demo.mid"
        refined_midi.write(output_path)
        print(f"\nRefined MIDI saved to: {output_path}")
    else:
        print("\nWarning: No babySlakh data found. Run from research/ directory.")
