# -*- coding: utf-8 -*-
"""
Decay-Based Consolidation Algorithm for Phase 1 MR-MT3 Enhancement

Reduces instrument leakage by merging instruments with similar:
- Decay characteristics (via Prony analysis)
- Spectral characteristics (via VQT spectral centroids)
- MIDI instrument families (optional constraint)

Implements greedy clustering with graceful degradation when Prony analysis fails.
"""

import numpy as np
import pretty_midi
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

from .config import EnhancementConfig


@dataclass
class MergeCandidate:
    """Represents a potential instrument merge"""
    inst_a: int  # First instrument index
    inst_b: int  # Second instrument index
    decay_similarity: float  # [0-1] similarity score
    spectral_diff: float  # Hz difference
    midi_class_match: bool  # Same MIDI family
    score: float  # Combined merge score


class InstrumentConsolidator:
    """
    Consolidate MIDI instruments using decay and spectral similarity

    Algorithm:
    1. Compute pairwise similarity between all instrument pairs
    2. Greedily merge most similar pairs that meet all criteria
    3. Iterate until no more valid merges exist

    Merging criteria (ALL must be met):
    - Same MIDI class (program // 8) if config requires it
    - Decay similarity > threshold (e.g., 0.80)
    - Spectral difference < threshold (e.g., 200 Hz)
    - Neither is percussion (is_drum == False)
    """

    def __init__(self, config: EnhancementConfig):
        """
        Initialize consolidator with configuration

        Args:
            config: EnhancementConfig with consolidation parameters
        """
        self.config = config
        self.consolidation_cfg = config.consolidation

        # Get active thresholds based on strategy
        thresholds = self.consolidation_cfg.get_active_thresholds()
        self.decay_threshold = thresholds["decay"]
        self.spectral_threshold = thresholds["spectral"]

        self.verbose = config.verbose
        self.debug = config.debug

    def compute_decay_similarity(
        self,
        decay_a: float,
        decay_b: float
    ) -> float:
        """
        Compute decay rate similarity between two instruments

        Uses normalized absolute difference scaled to [0, 1] where:
        - 1.0 = identical decay rates
        - 0.0 = maximally different decay rates

        Args:
            decay_a: First instrument's mean decay rate (negative value)
            decay_b: Second instrument's mean decay rate (negative value)

        Returns:
            Similarity score in [0, 1]
        """
        # Handle None or invalid values
        if decay_a is None or decay_b is None:
            return 0.0

        if not np.isfinite(decay_a) or not np.isfinite(decay_b):
            return 0.0

        # Both decay rates should be negative (damping coefficients)
        # More negative = faster decay
        decay_a = abs(decay_a)
        decay_b = abs(decay_b)

        # Avoid division by zero
        if decay_a == 0 and decay_b == 0:
            return 1.0

        # Compute relative difference
        max_decay = max(decay_a, decay_b)
        min_decay = min(decay_a, decay_b)

        if max_decay == 0:
            return 0.0

        # Similarity = 1 - relative_difference
        similarity = 1.0 - (max_decay - min_decay) / max_decay

        return float(np.clip(similarity, 0.0, 1.0))

    def compute_spectral_similarity(
        self,
        centroid_a: float,
        centroid_b: float
    ) -> float:
        """
        Compute spectral centroid difference between two instruments

        Args:
            centroid_a: First instrument's mean spectral centroid (Hz)
            centroid_b: Second instrument's mean spectral centroid (Hz)

        Returns:
            Absolute difference in Hz
        """
        # Handle None or invalid values
        if centroid_a is None or centroid_b is None:
            return float('inf')

        if not np.isfinite(centroid_a) or not np.isfinite(centroid_b):
            return float('inf')

        # Return absolute difference in Hz
        return float(abs(centroid_a - centroid_b))

    def get_midi_class(self, program: int) -> int:
        """
        Get MIDI instrument class (family) from program number

        MIDI program numbers are grouped into families of 8:
        - 0-7: Piano family
        - 8-15: Chromatic Percussion
        - 24-31: Guitar family
        - 32-39: Bass family
        - etc.

        Args:
            program: MIDI program number [0-127]

        Returns:
            MIDI class [0-15]
        """
        return program // 8

    def should_merge(
        self,
        features_a: Dict,
        features_b: Dict
    ) -> Tuple[bool, Optional[MergeCandidate]]:
        """
        Determine if two instruments should be merged

        Checks all merging criteria:
        1. Neither is percussion
        2. Same MIDI class if required by config
        3. Decay similarity exceeds threshold
        4. Spectral difference below threshold

        Args:
            features_a: Feature dict for first instrument
            features_b: Feature dict for second instrument

        Returns:
            Tuple of (should_merge, merge_candidate)
            merge_candidate is None if should_merge is False
        """
        # Extract feature values
        inst_a_idx = features_a.get('index', -1)
        inst_b_idx = features_b.get('index', -1)

        # 1. Check percussion exclusion
        if features_a.get('is_drum', False) or features_b.get('is_drum', False):
            if self.debug:
                print(f"  Skip: Instrument {inst_a_idx} or {inst_b_idx} is percussion")
            return False, None

        # 2. Check MIDI class matching if required
        program_a = features_a.get('program', 0)
        program_b = features_b.get('program', 0)

        class_a = self.get_midi_class(program_a)
        class_b = self.get_midi_class(program_b)
        midi_class_match = (class_a == class_b)

        if self.consolidation_cfg.midi_class_match_required and not midi_class_match:
            if self.debug:
                print(f"  Skip: MIDI class mismatch ({class_a} vs {class_b})")
            return False, None

        # 3. Check decay similarity
        decay_a = features_a.get('mean_decay')
        decay_b = features_b.get('mean_decay')

        if decay_a is None or decay_b is None:
            if self.debug:
                print(f"  Skip: Missing decay data")
            return False, None

        decay_similarity = self.compute_decay_similarity(decay_a, decay_b)

        if decay_similarity < self.decay_threshold:
            if self.debug:
                print(f"  Skip: Decay similarity {decay_similarity:.2f} < {self.decay_threshold:.2f}")
            return False, None

        # 4. Check spectral similarity
        centroid_a = features_a.get('mean_centroid')
        centroid_b = features_b.get('mean_centroid')

        if centroid_a is None or centroid_b is None:
            if self.debug:
                print(f"  Skip: Missing spectral data")
            return False, None

        spectral_diff = self.compute_spectral_similarity(centroid_a, centroid_b)

        if spectral_diff > self.spectral_threshold:
            if self.debug:
                print(f"  Skip: Spectral diff {spectral_diff:.1f} Hz > {self.spectral_threshold:.1f} Hz")
            return False, None

        # All criteria met - create merge candidate
        # Compute combined score (higher = better match)
        # Weight decay similarity more heavily than spectral similarity
        score = (decay_similarity * 0.7) + ((1.0 - spectral_diff / 1000.0) * 0.3)

        candidate = MergeCandidate(
            inst_a=inst_a_idx,
            inst_b=inst_b_idx,
            decay_similarity=decay_similarity,
            spectral_diff=spectral_diff,
            midi_class_match=midi_class_match,
            score=score
        )

        return True, candidate

    def merge_notes(
        self,
        instruments: List[pretty_midi.Instrument],
        inst_a_idx: int,
        inst_b_idx: int
    ) -> pretty_midi.Instrument:
        """
        Merge notes from two instruments into a single instrument

        Combines all notes from both instruments and sorts by start time.
        Takes metadata (name, program) from the instrument with more notes.

        Args:
            instruments: List of pretty_midi.Instrument objects
            inst_a_idx: Index of first instrument to merge
            inst_b_idx: Index of second instrument to merge

        Returns:
            New merged pretty_midi.Instrument
        """
        inst_a = instruments[inst_a_idx]
        inst_b = instruments[inst_b_idx]

        # Determine primary instrument (most notes)
        if len(inst_a.notes) >= len(inst_b.notes):
            primary = inst_a
            secondary = inst_b
        else:
            primary = inst_b
            secondary = inst_a

        # Create new instrument with primary's metadata
        merged = pretty_midi.Instrument(
            program=primary.program,
            is_drum=primary.is_drum,
            name=f"{primary.name or 'Unknown'}+{secondary.name or 'Unknown'}"
        )

        # Combine all notes
        all_notes = list(inst_a.notes) + list(inst_b.notes)

        # Sort by start time
        all_notes.sort(key=lambda n: n.start)

        # Add to merged instrument
        merged.notes = all_notes

        # Copy control changes from primary instrument
        merged.control_changes = list(primary.control_changes)
        merged.pitch_bends = list(primary.pitch_bends)

        return merged

    def consolidate(
        self,
        midi: pretty_midi.PrettyMIDI,
        features: Dict[int, Dict]
    ) -> Tuple[pretty_midi.PrettyMIDI, Dict]:
        """
        Main consolidation algorithm using greedy clustering

        Algorithm:
        1. Compute all pairwise merge candidates
        2. Sort by merge score (best first)
        3. Greedily merge best candidate
        4. Recompute features for merged instrument
        5. Repeat until no valid merges remain

        Args:
            midi: Input pretty_midi.PrettyMIDI object
            features: Dictionary mapping instrument index to features
                     (output from MIDIAwareFeatureExtractor)

        Returns:
            Tuple of (consolidated_midi, statistics_dict)
        """
        # Add index to features for tracking
        for idx, feat in features.items():
            feat['index'] = idx

        # Track merges
        merge_history = []
        iteration = 0
        max_iterations = 100  # Safety limit

        # Working copy of instruments
        current_instruments = list(midi.instruments)
        current_features = dict(features)

        if self.verbose:
            print(f"\nStarting consolidation with {len(current_instruments)} instruments")
            print(f"Strategy: {self.consolidation_cfg.strategy}")
            print(f"Thresholds: decay={self.decay_threshold:.2f}, spectral={self.spectral_threshold:.1f} Hz")

        while iteration < max_iterations:
            iteration += 1

            # Find all valid merge candidates
            candidates = []

            for i in range(len(current_instruments)):
                for j in range(i + 1, len(current_instruments)):
                    if i not in current_features or j not in current_features:
                        continue

                    should_merge, candidate = self.should_merge(
                        current_features[i],
                        current_features[j]
                    )

                    if should_merge and candidate is not None:
                        candidates.append(candidate)

            # If no candidates, we're done
            if not candidates:
                if self.verbose:
                    print(f"\nConverged after {iteration} iterations - no more valid merges")
                break

            # Sort by score (best first)
            candidates.sort(key=lambda c: c.score, reverse=True)
            best = candidates[0]

            if self.verbose:
                print(f"\nIteration {iteration}: Merging instruments {best.inst_a} + {best.inst_b}")
                print(f"  - Decay similarity: {best.decay_similarity:.3f}")
                print(f"  - Spectral diff: {best.spectral_diff:.1f} Hz")
                print(f"  - Score: {best.score:.3f}")

            # Perform merge
            merged_inst = self.merge_notes(
                current_instruments,
                best.inst_a,
                best.inst_b
            )

            # Update instrument list - remove originals, add merged
            # Build new list excluding merged indices
            new_instruments = []
            new_features = {}
            new_idx = 0

            for old_idx, inst in enumerate(current_instruments):
                if old_idx == best.inst_a or old_idx == best.inst_b:
                    continue  # Skip merged instruments

                new_instruments.append(inst)
                new_features[new_idx] = current_features[old_idx]
                new_features[new_idx]['index'] = new_idx
                new_idx += 1

            # Add merged instrument
            new_instruments.append(merged_inst)

            # Create features for merged instrument
            # Average the features from both parents
            merged_features = {
                'index': new_idx,
                'mean_decay': np.mean([
                    current_features[best.inst_a]['mean_decay'],
                    current_features[best.inst_b]['mean_decay']
                ]),
                'mean_centroid': np.mean([
                    current_features[best.inst_a]['mean_centroid'],
                    current_features[best.inst_b]['mean_centroid']
                ]),
                'mean_attack': np.mean([
                    current_features[best.inst_a].get('mean_attack', 0),
                    current_features[best.inst_b].get('mean_attack', 0)
                ]),
                'n_notes': len(merged_inst.notes),
                'program': merged_inst.program,
                'is_drum': merged_inst.is_drum,
                'prony_success_rate': np.mean([
                    current_features[best.inst_a].get('prony_success_rate', 0),
                    current_features[best.inst_b].get('prony_success_rate', 0)
                ])
            }

            new_features[new_idx] = merged_features

            # Update for next iteration
            current_instruments = new_instruments
            current_features = new_features

            # Record merge
            merge_history.append({
                'iteration': iteration,
                'merged': (best.inst_a, best.inst_b),
                'decay_similarity': best.decay_similarity,
                'spectral_diff': best.spectral_diff,
                'score': best.score
            })

        # Create output MIDI
        output_midi = pretty_midi.PrettyMIDI()
        output_midi.instruments = current_instruments

        # Compute statistics
        stats = {
            'input_instruments': len(midi.instruments),
            'output_instruments': len(current_instruments),
            'merges_performed': len(merge_history),
            'reduction_rate': 1.0 - (len(current_instruments) / len(midi.instruments)),
            'iterations': iteration,
            'merge_history': merge_history,
            'strategy': self.consolidation_cfg.strategy,
            'decay_threshold': self.decay_threshold,
            'spectral_threshold': self.spectral_threshold
        }

        if self.verbose:
            print(f"\nConsolidation complete:")
            print(f"  - Input: {stats['input_instruments']} instruments")
            print(f"  - Output: {stats['output_instruments']} instruments")
            print(f"  - Reduction: {stats['reduction_rate']:.1%}")

        return output_midi, stats

    def consolidate_with_fallback(
        self,
        midi: pretty_midi.PrettyMIDI,
        features: Dict[int, Dict]
    ) -> Tuple[pretty_midi.PrettyMIDI, Dict]:
        """
        Consolidate with graceful degradation for Prony failures

        If Prony analysis success rate is below threshold (<20%), disable
        decay-based merging and fall back to spectral-only merging.

        Args:
            midi: Input pretty_midi.PrettyMIDI object
            features: Feature dictionary from MIDIAwareFeatureExtractor

        Returns:
            Tuple of (consolidated_midi, statistics_dict)
        """
        # Compute average Prony success rate
        success_rates = [
            f.get('prony_success_rate', 0.0)
            for f in features.values()
        ]

        if not success_rates:
            if self.verbose:
                print("Warning: No features available, returning original MIDI")
            return midi, {'status': 'no_features', 'input_instruments': len(midi.instruments)}

        avg_success = np.mean(success_rates)

        if self.verbose:
            print(f"\nProny success rate: {avg_success:.1%}")

        # Check if we need fallback
        if avg_success < self.consolidation_cfg.prony_failure_threshold:
            if self.verbose:
                print(f"Warning: Prony success rate below threshold ({self.consolidation_cfg.prony_failure_threshold:.0%})")
                print("Falling back to spectral-only merging")

            # Disable decay-based merging by setting threshold to 0
            original_threshold = self.decay_threshold
            self.decay_threshold = 0.0

            # Set all decay values to neutral to disable decay checking
            for feat in features.values():
                feat['mean_decay'] = -1.0  # Neutral decay value

            # Run consolidation
            output_midi, stats = self.consolidate(midi, features)

            # Restore original threshold
            self.decay_threshold = original_threshold

            # Mark as fallback in stats
            stats['fallback_mode'] = True
            stats['prony_success_rate'] = avg_success

            return output_midi, stats

        # Normal consolidation
        output_midi, stats = self.consolidate(midi, features)
        stats['fallback_mode'] = False
        stats['prony_success_rate'] = avg_success

        return output_midi, stats


def consolidate_by_decay(
    midi: pretty_midi.PrettyMIDI,
    features: Dict[int, Dict],
    config: Optional[EnhancementConfig] = None
) -> Tuple[pretty_midi.PrettyMIDI, Dict]:
    """
    Convenience function for decay-based consolidation

    Args:
        midi: Input pretty_midi.PrettyMIDI object
        features: Feature dictionary from MIDIAwareFeatureExtractor
        config: EnhancementConfig (creates default if None)

    Returns:
        Tuple of (consolidated_midi, statistics_dict)
    """
    if config is None:
        config = EnhancementConfig()

    consolidator = InstrumentConsolidator(config)
    return consolidator.consolidate_with_fallback(midi, features)


if __name__ == "__main__":
    """Demo: Test consolidation with synthetic MIDI data"""

    print("=" * 70)
    print("Decay-Based Consolidation Demo")
    print("=" * 70)

    # Create synthetic MIDI with leakage
    print("\nCreating synthetic MIDI with simulated leakage...")

    midi = pretty_midi.PrettyMIDI()

    # Create 4 instruments that should merge into 2
    # Pair 1: Two piano-like instruments (similar decay, similar spectral)
    piano1 = pretty_midi.Instrument(program=0, name="Piano_1")
    piano2 = pretty_midi.Instrument(program=1, name="Piano_2_leaked")

    # Add notes to piano1
    for i, pitch in enumerate([60, 64, 67, 72]):
        note = pretty_midi.Note(
            velocity=80,
            pitch=pitch,
            start=i * 0.5,
            end=(i + 1) * 0.5
        )
        piano1.notes.append(note)

    # Add notes to piano2 (leaked from piano1)
    for i, pitch in enumerate([62, 65, 69]):
        note = pretty_midi.Note(
            velocity=60,
            pitch=pitch,
            start=i * 0.5 + 0.1,
            end=(i + 1) * 0.5 + 0.1
        )
        piano2.notes.append(note)

    # Pair 2: Two string instruments (similar characteristics)
    strings1 = pretty_midi.Instrument(program=40, name="Violin")
    strings2 = pretty_midi.Instrument(program=41, name="Viola_leaked")

    for i, pitch in enumerate([55, 59, 62]):
        note = pretty_midi.Note(
            velocity=70,
            pitch=pitch,
            start=i * 0.6,
            end=(i + 1) * 0.6
        )
        strings1.notes.append(note)

    for i, pitch in enumerate([57, 60]):
        note = pretty_midi.Note(
            velocity=50,
            pitch=pitch,
            start=i * 0.6 + 0.15,
            end=(i + 1) * 0.6 + 0.15
        )
        strings2.notes.append(note)

    midi.instruments = [piano1, piano2, strings1, strings2]

    print(f"Created MIDI with {len(midi.instruments)} instruments:")
    for i, inst in enumerate(midi.instruments):
        print(f"  {i}: {inst.name} (program={inst.program}, notes={len(inst.notes)})")

    # Create synthetic features
    print("\nCreating synthetic features (simulating similar decay/spectral)...")

    features = {
        0: {  # Piano 1
            'mean_decay': -2.5,
            'mean_centroid': 800.0,
            'mean_attack': 0.005,
            'n_notes': len(piano1.notes),
            'program': 0,
            'is_drum': False,
            'prony_success_rate': 0.95
        },
        1: {  # Piano 2 (leaked) - similar features
            'mean_decay': -2.6,  # Very similar decay
            'mean_centroid': 820.0,  # Close spectral centroid
            'mean_attack': 0.006,
            'n_notes': len(piano2.notes),
            'program': 1,
            'is_drum': False,
            'prony_success_rate': 0.90
        },
        2: {  # Strings 1
            'mean_decay': -1.8,
            'mean_centroid': 1200.0,
            'mean_attack': 0.015,
            'n_notes': len(strings1.notes),
            'program': 40,
            'is_drum': False,
            'prony_success_rate': 0.92
        },
        3: {  # Strings 2 (leaked) - similar features
            'mean_decay': -1.9,  # Similar decay
            'mean_centroid': 1180.0,  # Close spectral
            'mean_attack': 0.014,
            'n_notes': len(strings2.notes),
            'program': 41,
            'is_drum': False,
            'prony_success_rate': 0.88
        }
    }

    # Test different strategies
    for strategy in ["conservative", "balanced", "aggressive"]:
        print(f"\n{'-' * 70}")
        print(f"Testing {strategy.upper()} strategy")
        print(f"{'-' * 70}")

        config = EnhancementConfig(verbose=True, debug=False)
        config.consolidation.strategy = strategy

        # Run consolidation
        consolidated, stats = consolidate_by_decay(midi, features, config)

        print(f"\nResults ({strategy}):")
        print(f"  - Merges performed: {stats['merges_performed']}")
        print(f"  - Final instruments: {stats['output_instruments']}")
        print(f"  - Reduction: {stats['reduction_rate']:.1%}")

        if stats['merge_history']:
            print(f"\n  Merge details:")
            for merge in stats['merge_history']:
                print(f"    Iteration {merge['iteration']}: {merge['merged']}")
                print(f"      -> Decay sim: {merge['decay_similarity']:.3f}")
                print(f"      -> Spectral diff: {merge['spectral_diff']:.1f} Hz")

    # Test fallback mode with low Prony success
    print(f"\n{'-' * 70}")
    print("Testing fallback mode (low Prony success)")
    print(f"{'-' * 70}")

    # Modify features to simulate Prony failure
    low_success_features = dict(features)
    for feat in low_success_features.values():
        feat['prony_success_rate'] = 0.15  # Below 20% threshold

    config = EnhancementConfig(verbose=True)
    consolidated, stats = consolidate_by_decay(midi, low_success_features, config)

    print(f"\nFallback mode results:")
    print(f"  - Fallback activated: {stats['fallback_mode']}")
    print(f"  - Prony success: {stats['prony_success_rate']:.1%}")
    print(f"  - Final instruments: {stats['output_instruments']}")

    print("\n--- Demo complete!")
