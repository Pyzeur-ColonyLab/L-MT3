# -*- coding: utf-8 -*-
"""
MIDI-Aware Feature Extraction for Phase 1 MR-MT3 Enhancement

Wraps laplace_audio_analysis.py with MIDI-specific functionality:
- Per-note decay rate extraction
- Per-instrument spectral centroid computation
- Per-instrument attack time estimation

Designed to work with pretty_midi.PrettyMIDI objects and raw audio.
"""

import numpy as np
import librosa
import pretty_midi
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path

# Import base analyzers from parent directory
sys.path.append(str(Path(__file__).parent.parent))
from laplace_audio_analysis import PronyAnalyzer, VariableQWavelet, GammatoneFilterbank

from .config import EnhancementConfig


class MIDIAwareFeatureExtractor:
    """
    Extract Laplace features aligned with MIDI note events

    For each instrument in a MIDI file:
    - Extract decay rates per note using Prony analysis
    - Compute spectral centroids per note using VQT
    - Estimate attack times per note using Gammatone filterbank
    """

    def __init__(self, config: EnhancementConfig):
        """
        Initialize feature extractors with configuration

        Args:
            config: EnhancementConfig with Prony, VQT, Gammatone parameters
        """
        self.config = config
        self.sr = config.sample_rate

        # Initialize base analyzers
        self.prony = PronyAnalyzer(
            n_components=config.prony.n_components,
            model_order=config.prony.model_order
        )

        self.vqt = VariableQWavelet(
            fmin=config.vqt.fmin,
            n_bins=config.vqt.n_bins,
            bins_per_octave=config.vqt.bins_per_octave,
            q_rate=config.vqt.q_rate
        )

        self.gammatone = GammatoneFilterbank(
            n_filters=config.gammatone.n_filters,
            fmin=config.gammatone.fmin,
            fmax=config.gammatone.fmax
        )

    def extract_instrument_features(
        self,
        instrument: pretty_midi.Instrument,
        audio: np.ndarray
    ) -> Dict[str, any]:
        """
        Extract all features for a single instrument

        Args:
            instrument: pretty_midi.Instrument object with notes
            audio: Raw audio array at self.sr sample rate

        Returns:
            Dictionary with:
            - 'decay_rates': List of decay rates per note
            - 'spectral_centroids': List of spectral centroids per note
            - 'attack_times': List of attack times per note
            - 'mean_decay': Average decay rate (instrument signature)
            - 'mean_centroid': Average spectral centroid
            - 'mean_attack': Average attack time
            - 'prony_success_rate': Fraction of notes successfully analyzed
        """
        decay_rates = []
        spectral_centroids = []
        attack_times = []
        prony_successes = 0

        for note in instrument.notes:
            # Extract note segment from audio
            start_sample = int(note.start * self.sr)
            end_sample = int(note.end * self.sr)

            # Ensure valid bounds
            if end_sample > len(audio):
                end_sample = len(audio)
            if start_sample >= end_sample:
                continue

            note_audio = audio[start_sample:end_sample]
            note_duration = (end_sample - start_sample) / self.sr

            # Skip very short notes
            if note_duration < self.config.prony.min_note_duration:
                continue

            # 1. Extract decay rate via Prony analysis
            decay = self._extract_note_decay(note_audio, note.pitch)
            if decay is not None:
                decay_rates.append(decay)
                prony_successes += 1

            # 2. Extract spectral centroid via VQT
            centroid = self._extract_note_centroid(note_audio)
            if centroid is not None:
                spectral_centroids.append(centroid)

            # 3. Extract attack time via Gammatone
            attack = self._extract_note_attack(note_audio)
            if attack is not None:
                attack_times.append(attack)

        # Compute aggregate statistics
        prony_success_rate = prony_successes / max(len(instrument.notes), 1)

        return {
            'decay_rates': decay_rates,
            'spectral_centroids': spectral_centroids,
            'attack_times': attack_times,
            'mean_decay': np.median(decay_rates) if decay_rates else None,
            'mean_centroid': np.mean(spectral_centroids) if spectral_centroids else None,
            'mean_attack': np.median(attack_times) if attack_times else None,
            'prony_success_rate': prony_success_rate,
            'n_notes': len(instrument.notes),
            'program': instrument.program,
            'is_drum': instrument.is_drum
        }

    def _extract_note_decay(self, note_audio: np.ndarray, pitch: int) -> Optional[float]:
        """
        Extract decay rate for a single note using Prony analysis

        Args:
            note_audio: Audio segment for this note
            pitch: MIDI pitch number (for frequency reference)

        Returns:
            Median negative damping coefficient (decay rate), or None if failed
        """
        try:
            # Run Prony analysis on note segment
            result = self.prony.analyze(
                note_audio,
                self.sr,
                hop_length=self.config.prony.hop_length,
                win_length=self.config.prony.win_length
            )

            # Extract damping coefficients
            all_damping = [d for frame in result['damping'] for d in frame if d < 0]

            if len(all_damping) == 0:
                return None

            # Return median negative damping (decay rate)
            # More negative = faster decay
            return float(np.median(all_damping))

        except Exception as e:
            if self.config.debug:
                print(f"Prony analysis failed for note {pitch}: {e}")
            return None

    def _extract_note_centroid(self, note_audio: np.ndarray) -> Optional[float]:
        """
        Extract spectral centroid for a single note using VQT

        Args:
            note_audio: Audio segment for this note

        Returns:
            Mean spectral centroid in Hz, or None if failed
        """
        try:
            # Compute VQT
            vqt, times, freqs = self.vqt.analyze(note_audio, self.sr)
            vqt_mag = np.abs(vqt)

            # Compute spectral centroid across time
            # Weighted average of frequencies by magnitude
            centroids = []
            for t_idx in range(vqt_mag.shape[1]):
                frame = vqt_mag[:, t_idx]
                if np.sum(frame) > 0:
                    centroid = np.sum(freqs * frame) / np.sum(frame)
                    centroids.append(centroid)

            if len(centroids) == 0:
                return None

            # Return mean spectral centroid over note duration
            return float(np.mean(centroids))

        except Exception as e:
            if self.config.debug:
                print(f"VQT centroid extraction failed: {e}")
            return None

    def _extract_note_attack(self, note_audio: np.ndarray) -> Optional[float]:
        """
        Extract attack time for a single note using Gammatone envelope

        Args:
            note_audio: Audio segment for this note

        Returns:
            Attack time in seconds, or None if failed
        """
        try:
            # Apply Gammatone filterbank
            filtered, envelopes, center_freqs = self.gammatone.analyze(note_audio, self.sr)

            # Find channel with strongest response
            max_energies = np.max(envelopes, axis=1)
            dominant_channel = np.argmax(max_energies)

            # Extract envelope from dominant channel
            envelope = envelopes[dominant_channel, :]

            # Estimate attack time: time to reach 90% of peak
            peak_value = np.max(envelope)
            threshold = 0.9 * peak_value

            # Find first time exceeding threshold
            above_threshold = np.where(envelope >= threshold)[0]
            if len(above_threshold) == 0:
                return None

            attack_samples = above_threshold[0]
            attack_time = attack_samples / self.sr

            return float(attack_time)

        except Exception as e:
            if self.config.debug:
                print(f"Gammatone attack extraction failed: {e}")
            return None

    def extract_all_features(
        self,
        midi: pretty_midi.PrettyMIDI,
        audio: np.ndarray
    ) -> Dict[int, Dict[str, any]]:
        """
        Extract features for all instruments in a MIDI file

        Args:
            midi: pretty_midi.PrettyMIDI object
            audio: Raw audio array at self.sr sample rate

        Returns:
            Dictionary mapping instrument index to feature dict
        """
        features = {}

        for i, instrument in enumerate(midi.instruments):
            if self.config.verbose:
                print(f"Extracting features for instrument {i} ({instrument.name or 'Unknown'})")

            inst_features = self.extract_instrument_features(instrument, audio)
            features[i] = inst_features

            if self.config.verbose:
                print(f"  - Notes: {inst_features['n_notes']}")
                print(f"  - Prony success: {inst_features['prony_success_rate']:.1%}")
                if inst_features['mean_decay'] is not None:
                    print(f"  - Mean decay: {inst_features['mean_decay']:.2f}")
                if inst_features['mean_centroid'] is not None:
                    print(f"  - Mean centroid: {inst_features['mean_centroid']:.1f} Hz")

        return features

    def compute_harmonic_ratio(self, note_audio: np.ndarray) -> float:
        """
        Compute harmonic-to-noise ratio for a note segment

        Used in refinement heuristics to distinguish harmonic instruments
        from percussion/noise-based sounds.

        Args:
            note_audio: Audio segment

        Returns:
            Harmonic ratio (higher = more harmonic)
        """
        try:
            # Use librosa's harmonic-percussive separation
            harmonic, percussive = librosa.effects.hpss(note_audio)

            # Compute energy ratio
            harmonic_energy = np.sum(harmonic ** 2)
            percussive_energy = np.sum(percussive ** 2)

            if percussive_energy == 0:
                return float('inf')

            ratio = harmonic_energy / percussive_energy
            return float(ratio)

        except Exception as e:
            if self.config.debug:
                print(f"Harmonic ratio computation failed: {e}")
            return 1.0  # Neutral value


def extract_features_from_midi(
    midi: pretty_midi.PrettyMIDI,
    audio: np.ndarray,
    sr: int = 16000,
    config: Optional[EnhancementConfig] = None
) -> Dict[int, Dict[str, any]]:
    """
    Convenience function to extract features from MIDI + audio

    Args:
        midi: pretty_midi.PrettyMIDI object
        audio: Raw audio array
        sr: Sample rate (default 16000 for MR-MT3)
        config: Enhancement configuration (creates default if None)

    Returns:
        Dictionary mapping instrument index to feature dict
    """
    if config is None:
        config = EnhancementConfig()
        config.sample_rate = sr

    extractor = MIDIAwareFeatureExtractor(config)
    return extractor.extract_all_features(midi, audio)


if __name__ == "__main__":
    # Demo: Load babySlakh track and extract features
    import librosa

    print("=" * 70)
    print("MIDI-Aware Feature Extraction Demo")
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
        print(f"MIDI: {len(midi.instruments)} instruments, {midi.get_end_time():.1f}s")

        # Extract features
        config = EnhancementConfig(verbose=True)
        features = extract_features_from_midi(midi, audio, sr, config)

        print(f"\n✓ Extracted features for {len(features)} instruments")
    else:
        print("\n⚠️  No babySlakh data found. Run from research/ directory.")
