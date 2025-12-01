"""
Feature Extraction for NSynth Dataset
======================================

Extracts 26 Laplace features from NSynth audio samples
for training the instrument classifier.

Features:
- 7 Prony features (decay characteristics)
- 8 VQT features (spectral characteristics)  
- 11 Gammatone features (perceptual/envelope)

Usage:
    python extract_nsynth_features.py \
        --nsynth_dir ./nsynth-train \
        --output ./nsynth_features.npz \
        --max_samples 10000

Author: Aurel (ColonyLab)
Date: November 2025
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Audio processing
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    warnings.warn("librosa not available")

# Scipy for signal processing
from scipy import signal
from scipy.linalg import lstsq, toeplitz
from scipy.signal import hilbert


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ExtractionConfig:
    """Configuration for feature extraction"""
    
    # Audio parameters
    sample_rate: int = 16000
    
    # Prony parameters
    prony_n_components: int = 10
    prony_model_order: int = 30
    prony_window_size: int = 1024
    prony_hop_size: int = 256
    
    # VQT parameters
    vqt_fmin: float = 55.0  # A1
    vqt_n_bins: int = 72    # 6 octaves
    vqt_bins_per_octave: int = 12
    
    # Gammatone parameters
    gammatone_n_filters: int = 48
    gammatone_fmin: float = 50.0
    gammatone_fmax: float = 4000.0
    
    # Processing
    n_workers: int = 4
    batch_size: int = 100


# =============================================================================
# PRONY ANALYSIS
# =============================================================================

class PronyAnalyzer:
    """
    Prony Analysis for extracting decay characteristics.
    
    Models signal as sum of exponentially damped sinusoids:
    x[n] = Σ A_k * exp(σ_k * n) * exp(j * ω_k * n)
    
    Where:
    - σ_k: damping factor (decay rate)
    - ω_k: angular frequency
    - A_k: amplitude
    """
    
    def __init__(self, n_components: int = 10, model_order: int = 30):
        self.n_components = n_components
        self.model_order = model_order
    
    def analyze(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Perform Prony analysis on audio segment.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary with Prony features
        """
        try:
            # Use shorter segment for stability
            segment = audio[:min(len(audio), 4096)]
            
            # Normalize
            segment = segment / (np.max(np.abs(segment)) + 1e-8)
            
            # Prony decomposition
            dampings, frequencies, amplitudes = self._prony_decomposition(segment, sr)
            
            if len(dampings) == 0:
                return self._default_features()
            
            # Sort by amplitude (most significant components first)
            sort_idx = np.argsort(amplitudes)[::-1]
            dampings = dampings[sort_idx]
            frequencies = frequencies[sort_idx]
            amplitudes = amplitudes[sort_idx]
            
            # Take top components
            n_keep = min(self.n_components, len(dampings))
            dampings = dampings[:n_keep]
            frequencies = frequencies[:n_keep]
            amplitudes = amplitudes[:n_keep]
            
            # Compute features
            features = {
                'prony_mean_damping': float(np.mean(dampings)),
                'prony_std_damping': float(np.std(dampings)) if len(dampings) > 1 else 0.0,
                'prony_median_damping': float(np.median(dampings)),
                'prony_damping_range': float(np.max(dampings) - np.min(dampings)) if len(dampings) > 1 else 0.0,
                'prony_mean_freq': float(np.mean(frequencies)),
                'prony_freq_spread': float(np.std(frequencies)) if len(frequencies) > 1 else 0.0,
                'prony_spectral_centroid': float(np.sum(frequencies * amplitudes) / (np.sum(amplitudes) + 1e-8))
            }
            
            return features
            
        except Exception as e:
            warnings.warn(f"Prony analysis failed: {e}")
            return self._default_features()
    
    def _prony_decomposition(self, signal_data: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform Prony decomposition using linear prediction.
        """
        n = len(signal_data)
        p = min(self.model_order, n // 3)
        
        if p < 2:
            return np.array([]), np.array([]), np.array([])
        
        # Build Toeplitz matrix for linear prediction
        # y = [x[p], x[p+1], ..., x[n-1]]
        # A = toeplitz of x values
        
        y = signal_data[p:]
        
        # Create data matrix
        A = np.zeros((n - p, p))
        for i in range(n - p):
            A[i, :] = signal_data[i:i+p][::-1]
        
        # Solve least squares: A @ a = y
        try:
            coeffs, _, _, _ = lstsq(A, y)
        except:
            return np.array([]), np.array([]), np.array([])
        
        # Find roots of characteristic polynomial
        # z^p - a[0]*z^(p-1) - ... - a[p-1] = 0
        poly_coeffs = np.concatenate([[1], -coeffs])
        
        try:
            roots = np.roots(poly_coeffs)
        except:
            return np.array([]), np.array([]), np.array([])
        
        # Filter valid roots (inside or on unit circle, positive frequency)
        valid_mask = (np.abs(roots) <= 1.0) & (np.abs(roots) > 0.01)
        roots = roots[valid_mask]
        
        if len(roots) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Convert to damping and frequency
        # z = exp((σ + jω) * T) where T = 1/sr
        dampings = np.log(np.abs(roots)) * sr  # In 1/s
        frequencies = np.angle(roots) * sr / (2 * np.pi)  # In Hz
        
        # Only keep positive frequencies
        pos_mask = frequencies > 0
        dampings = dampings[pos_mask]
        frequencies = frequencies[pos_mask]
        
        if len(dampings) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Estimate amplitudes using least squares
        n_comp = len(dampings)
        t = np.arange(len(signal_data)) / sr
        
        # Build basis matrix
        basis = np.zeros((len(signal_data), n_comp), dtype=complex)
        for i, (d, f) in enumerate(zip(dampings, frequencies)):
            basis[:, i] = np.exp((d + 2j * np.pi * f) * t)
        
        try:
            amplitudes, _, _, _ = lstsq(basis, signal_data)
            amplitudes = np.abs(amplitudes)
        except:
            amplitudes = np.ones(n_comp)
        
        return dampings, frequencies, amplitudes
    
    def _default_features(self) -> Dict:
        """Return default features when analysis fails."""
        return {
            'prony_mean_damping': -1000.0,
            'prony_std_damping': 500.0,
            'prony_median_damping': -1000.0,
            'prony_damping_range': 1000.0,
            'prony_mean_freq': 440.0,
            'prony_freq_spread': 200.0,
            'prony_spectral_centroid': 500.0
        }


# =============================================================================
# VQT ANALYSIS
# =============================================================================

class VQTAnalyzer:
    """
    Variable-Q Transform for spectral analysis.
    
    VQT provides better frequency resolution at low frequencies
    and better time resolution at high frequencies.
    """
    
    def __init__(self, fmin: float = 55.0, n_bins: int = 72, bins_per_octave: int = 12):
        self.fmin = fmin
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
    
    def analyze(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Compute VQT features.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary with VQT features
        """
        try:
            # Compute VQT
            vqt = librosa.vqt(
                audio,
                sr=sr,
                fmin=self.fmin,
                n_bins=self.n_bins,
                bins_per_octave=self.bins_per_octave
            )
            
            # Magnitude
            vqt_mag = np.abs(vqt)
            
            # Frequency bins
            freqs = librosa.cqt_frequencies(
                n_bins=self.n_bins,
                fmin=self.fmin,
                bins_per_octave=self.bins_per_octave
            )
            
            # Time-averaged spectrum
            mean_spectrum = np.mean(vqt_mag, axis=1)
            
            # Spectral centroid
            centroid = np.sum(freqs * mean_spectrum) / (np.sum(mean_spectrum) + 1e-8)
            
            # Spectral spread (standard deviation)
            spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * mean_spectrum) / (np.sum(mean_spectrum) + 1e-8))
            
            # Spectral skewness
            skewness = np.sum(((freqs - centroid) ** 3) * mean_spectrum) / ((spread ** 3 + 1e-8) * (np.sum(mean_spectrum) + 1e-8))
            
            # Temporal variation
            temporal_var = np.std(vqt_mag, axis=1)
            
            # Harmonic ratio (ratio of harmonic to total energy)
            # Simplified: ratio of peaks to total
            peaks = signal.find_peaks(mean_spectrum, distance=3)[0]
            if len(peaks) > 0:
                harmonic_energy = np.sum(mean_spectrum[peaks])
                total_energy = np.sum(mean_spectrum)
                harmonic_ratio = harmonic_energy / (total_energy + 1e-8)
            else:
                harmonic_ratio = 0.5
            
            # Phase coherence (simplified)
            phase = np.angle(vqt)
            phase_diff = np.diff(phase, axis=1)
            phase_coherence = 1.0 - np.mean(np.std(phase_diff, axis=1)) / np.pi
            
            # Damping estimate from temporal decay
            energy_over_time = np.sum(vqt_mag ** 2, axis=0)
            if len(energy_over_time) > 10:
                # Fit exponential decay
                t = np.arange(len(energy_over_time))
                log_energy = np.log(energy_over_time + 1e-8)
                # Linear fit: log(E) = a*t + b => damping = a * sr / hop
                try:
                    slope, _ = np.polyfit(t[:len(t)//2], log_energy[:len(t)//2], 1)
                    damping_estimate = slope * sr / 512  # Approximate hop size
                except:
                    damping_estimate = -1000.0
            else:
                damping_estimate = -1000.0
            
            return {
                'vqt_spectral_centroid': float(centroid),
                'vqt_spectral_spread': float(spread),
                'vqt_spectral_skewness': float(np.clip(skewness, -10, 10)),
                'vqt_temporal_var_mean': float(np.mean(temporal_var)),
                'vqt_temporal_var_max': float(np.max(temporal_var)),
                'vqt_harmonic_ratio': float(harmonic_ratio),
                'vqt_phase_coherence': float(np.clip(phase_coherence, 0, 1)),
                'vqt_damping_estimate': float(np.clip(damping_estimate, -10000, 0))
            }
            
        except Exception as e:
            warnings.warn(f"VQT analysis failed: {e}")
            return self._default_features()
    
    def _default_features(self) -> Dict:
        """Return default features when analysis fails."""
        return {
            'vqt_spectral_centroid': 500.0,
            'vqt_spectral_spread': 200.0,
            'vqt_spectral_skewness': 0.0,
            'vqt_temporal_var_mean': 0.1,
            'vqt_temporal_var_max': 0.3,
            'vqt_harmonic_ratio': 0.5,
            'vqt_phase_coherence': 0.5,
            'vqt_damping_estimate': -1000.0
        }


# =============================================================================
# GAMMATONE ANALYSIS
# =============================================================================

class GammatoneAnalyzer:
    """
    Gammatone filterbank analysis for perceptual features.
    
    Gammatone filters model the human auditory system and are
    useful for extracting perceptually-relevant features like
    attack time and envelope characteristics.
    """
    
    def __init__(self, n_filters: int = 48, fmin: float = 50.0, fmax: float = 4000.0):
        self.n_filters = n_filters
        self.fmin = fmin
        self.fmax = fmax
    
    def analyze(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Compute Gammatone features.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary with Gammatone features
        """
        try:
            # Create ERB-spaced center frequencies
            erb_min = self._hz_to_erb(self.fmin)
            erb_max = self._hz_to_erb(self.fmax)
            erb_points = np.linspace(erb_min, erb_max, self.n_filters)
            center_freqs = self._erb_to_hz(erb_points)
            
            # Apply gammatone filterbank
            filtered_signals = []
            for cf in center_freqs:
                filtered = self._gammatone_filter(audio, cf, sr)
                filtered_signals.append(filtered)
            
            filtered_signals = np.array(filtered_signals)
            
            # Extract envelopes using Hilbert transform
            envelopes = np.abs(hilbert(filtered_signals, axis=1))
            
            # Compute features from envelopes
            
            # Attack time (time to reach 90% of max)
            attack_times = []
            for env in envelopes:
                max_val = np.max(env)
                threshold = 0.9 * max_val
                above_threshold = np.where(env >= threshold)[0]
                if len(above_threshold) > 0:
                    attack_time = above_threshold[0] / sr
                else:
                    attack_time = len(env) / sr
                attack_times.append(attack_time)
            
            attack_times = np.array(attack_times)
            
            # Decay time (time to decay to 10% of max after peak)
            decay_times = []
            for env in envelopes:
                peak_idx = np.argmax(env)
                max_val = env[peak_idx]
                threshold = 0.1 * max_val
                post_peak = env[peak_idx:]
                below_threshold = np.where(post_peak <= threshold)[0]
                if len(below_threshold) > 0:
                    decay_time = below_threshold[0] / sr
                else:
                    decay_time = len(post_peak) / sr
                decay_times.append(decay_time)
            
            decay_times = np.array(decay_times)
            
            # Energy distribution across bands
            band_energies = np.sum(envelopes ** 2, axis=1)
            total_energy = np.sum(band_energies) + 1e-8
            
            # Divide into low/mid/high bands
            n_low = self.n_filters // 3
            n_mid = self.n_filters // 3
            
            low_energy = np.sum(band_energies[:n_low]) / total_energy
            mid_energy = np.sum(band_energies[n_low:n_low+n_mid]) / total_energy
            high_energy = np.sum(band_energies[n_low+n_mid:]) / total_energy
            
            # Energy centroid and spread
            band_indices = np.arange(self.n_filters)
            energy_centroid = np.sum(band_indices * band_energies) / total_energy
            energy_spread = np.sqrt(np.sum(((band_indices - energy_centroid) ** 2) * band_energies) / total_energy)
            
            # Onset strength (how sharp is the attack)
            mean_env = np.mean(envelopes, axis=0)
            onset_diff = np.diff(mean_env)
            onset_strength = np.max(onset_diff) / (np.mean(np.abs(onset_diff)) + 1e-8)
            
            # Envelope flatness
            env_flatness = np.exp(np.mean(np.log(mean_env + 1e-8))) / (np.mean(mean_env) + 1e-8)
            
            return {
                'gt_mean_attack_time': float(np.mean(attack_times)),
                'gt_std_attack_time': float(np.std(attack_times)),
                'gt_mean_decay_time': float(np.mean(decay_times)),
                'gt_std_decay_time': float(np.std(decay_times)),
                'gt_energy_centroid': float(energy_centroid / self.n_filters),
                'gt_energy_spread': float(energy_spread / self.n_filters),
                'gt_low_energy_ratio': float(low_energy),
                'gt_mid_energy_ratio': float(mid_energy),
                'gt_high_energy_ratio': float(high_energy),
                'gt_onset_strength': float(np.clip(onset_strength, 0, 100)),
                'gt_envelope_flatness': float(np.clip(env_flatness, 0, 1))
            }
            
        except Exception as e:
            warnings.warn(f"Gammatone analysis failed: {e}")
            return self._default_features()
    
    def _hz_to_erb(self, hz: float) -> float:
        """Convert Hz to ERB scale."""
        return 21.4 * np.log10(1 + hz / 229)
    
    def _erb_to_hz(self, erb: np.ndarray) -> np.ndarray:
        """Convert ERB scale to Hz."""
        return 229 * (10 ** (erb / 21.4) - 1)
    
    def _gammatone_filter(self, audio: np.ndarray, cf: float, sr: int, order: int = 4) -> np.ndarray:
        """
        Apply a gammatone filter at the given center frequency.
        
        Simplified implementation using cascaded bandpass filters.
        """
        # ERB bandwidth
        erb = 24.7 * (4.37 * cf / 1000 + 1)
        b = 1.019 * 2 * np.pi * erb
        
        # Create bandpass filter
        low = max(cf - erb/2, 20) / (sr/2)
        high = min(cf + erb/2, sr/2 - 10) / (sr/2)
        
        if low >= high or low <= 0 or high >= 1:
            return audio
        
        try:
            sos = signal.butter(order, [low, high], btype='band', output='sos')
            filtered = signal.sosfilt(sos, audio)
            return filtered
        except:
            return audio
    
    def _default_features(self) -> Dict:
        """Return default features when analysis fails."""
        return {
            'gt_mean_attack_time': 0.05,
            'gt_std_attack_time': 0.02,
            'gt_mean_decay_time': 0.3,
            'gt_std_decay_time': 0.1,
            'gt_energy_centroid': 0.5,
            'gt_energy_spread': 0.2,
            'gt_low_energy_ratio': 0.33,
            'gt_mid_energy_ratio': 0.34,
            'gt_high_energy_ratio': 0.33,
            'gt_onset_strength': 1.0,
            'gt_envelope_flatness': 0.5
        }


# =============================================================================
# COMBINED FEATURE EXTRACTOR
# =============================================================================

class LaplaceFeatureExtractor:
    """
    Combined feature extractor using Prony, VQT, and Gammatone analysis.
    
    Extracts 26 features total:
    - 7 Prony features
    - 8 VQT features
    - 11 Gammatone features
    """
    
    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        
        self.prony = PronyAnalyzer(
            n_components=self.config.prony_n_components,
            model_order=self.config.prony_model_order
        )
        
        self.vqt = VQTAnalyzer(
            fmin=self.config.vqt_fmin,
            n_bins=self.config.vqt_n_bins,
            bins_per_octave=self.config.vqt_bins_per_octave
        )
        
        self.gammatone = GammatoneAnalyzer(
            n_filters=self.config.gammatone_n_filters,
            fmin=self.config.gammatone_fmin,
            fmax=self.config.gammatone_fmax
        )
    
    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract all 26 features from audio.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Feature vector of shape (26,)
        """
        # Extract features from each analyzer
        prony_features = self.prony.analyze(audio, sr)
        vqt_features = self.vqt.analyze(audio, sr)
        gammatone_features = self.gammatone.analyze(audio, sr)
        
        # Combine into single vector (order matters!)
        feature_vector = np.array([
            # Prony (7)
            prony_features['prony_mean_damping'],
            prony_features['prony_std_damping'],
            prony_features['prony_median_damping'],
            prony_features['prony_damping_range'],
            prony_features['prony_mean_freq'],
            prony_features['prony_freq_spread'],
            prony_features['prony_spectral_centroid'],
            # VQT (8)
            vqt_features['vqt_spectral_centroid'],
            vqt_features['vqt_spectral_spread'],
            vqt_features['vqt_spectral_skewness'],
            vqt_features['vqt_temporal_var_mean'],
            vqt_features['vqt_temporal_var_max'],
            vqt_features['vqt_harmonic_ratio'],
            vqt_features['vqt_phase_coherence'],
            vqt_features['vqt_damping_estimate'],
            # Gammatone (11)
            gammatone_features['gt_mean_attack_time'],
            gammatone_features['gt_std_attack_time'],
            gammatone_features['gt_mean_decay_time'],
            gammatone_features['gt_std_decay_time'],
            gammatone_features['gt_energy_centroid'],
            gammatone_features['gt_energy_spread'],
            gammatone_features['gt_low_energy_ratio'],
            gammatone_features['gt_mid_energy_ratio'],
            gammatone_features['gt_high_energy_ratio'],
            gammatone_features['gt_onset_strength'],
            gammatone_features['gt_envelope_flatness'],
        ], dtype=np.float32)
        
        return feature_vector
    
    def extract_dict(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract features as dictionary.
        
        Returns all features with their names.
        """
        prony_features = self.prony.analyze(audio, sr)
        vqt_features = self.vqt.analyze(audio, sr)
        gammatone_features = self.gammatone.analyze(audio, sr)
        
        return {**prony_features, **vqt_features, **gammatone_features}


# =============================================================================
# NSYNTH DATASET PROCESSING
# =============================================================================

def process_single_nsynth_sample(args: Tuple[str, str, int]) -> Optional[Tuple[np.ndarray, int, str]]:
    """
    Process a single NSynth sample.
    
    Args:
        args: Tuple of (audio_path, note_id, instrument_family)
        
    Returns:
        Tuple of (features, label, note_id) or None if failed
    """
    audio_path, note_id, instrument_family = args
    
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Extract features
        extractor = LaplaceFeatureExtractor()
        features = extractor.extract(audio, sr)
        
        return (features, instrument_family, note_id)
        
    except Exception as e:
        return None


def extract_nsynth_features(
    nsynth_dir: str,
    output_path: str,
    max_samples: Optional[int] = None,
    n_workers: int = 4,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract Laplace features from entire NSynth dataset.
    
    Args:
        nsynth_dir: Path to NSynth directory (containing audio/ and examples.json)
        output_path: Path to save features (.npz file)
        max_samples: Maximum samples to process (None for all)
        n_workers: Number of parallel workers
        verbose: Print progress
        
    Returns:
        Tuple of (features array, labels array)
    """
    nsynth_dir = Path(nsynth_dir)

    # Auto-detect correct directory structure
    # NSynth extraction may create nested directory: nsynth-train/nsynth-train/
    json_path = nsynth_dir / 'examples.json'
    if not json_path.exists():
        # Try nested directory
        nested_dir = nsynth_dir / nsynth_dir.name
        nested_json = nested_dir / 'examples.json'
        if nested_json.exists():
            nsynth_dir = nested_dir
            json_path = nested_json
            if verbose:
                print(f"Found nested structure, using: {nsynth_dir}")
        else:
            raise FileNotFoundError(
                f"examples.json not found in {nsynth_dir} or {nested_dir}\n"
                f"Expected structure: nsynth-train/examples.json or nsynth-train/nsynth-train/examples.json"
            )
    
    if verbose:
        print(f"Loading metadata from {json_path}...")
    
    with open(json_path) as f:
        metadata = json.load(f)
    
    if verbose:
        print(f"Found {len(metadata)} samples")
    
    # Prepare processing tasks
    tasks = []
    for note_id, info in metadata.items():
        audio_path = nsynth_dir / 'audio' / f"{note_id}.wav"
        if audio_path.exists():
            tasks.append((str(audio_path), note_id, info['instrument_family']))
    
    if max_samples is not None:
        tasks = tasks[:max_samples]
    
    if verbose:
        print(f"Processing {len(tasks)} samples with {n_workers} workers...")
    
    # Process in parallel
    features_list = []
    labels_list = []
    note_ids_list = []
    
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_single_nsynth_sample, task): task for task in tasks}
            
            for future in tqdm(as_completed(futures), total=len(futures), disable=not verbose):
                result = future.result()
                if result is not None:
                    features, label, note_id = result
                    features_list.append(features)
                    labels_list.append(label)
                    note_ids_list.append(note_id)
    else:
        # Sequential processing
        for task in tqdm(tasks, disable=not verbose):
            result = process_single_nsynth_sample(task)
            if result is not None:
                features, label, note_id = result
                features_list.append(features)
                labels_list.append(label)
                note_ids_list.append(note_id)
    
    # Convert to arrays
    features_array = np.array(features_list, dtype=np.float32)
    labels_array = np.array(labels_list, dtype=np.int32)
    
    if verbose:
        print(f"\nExtracted {len(features_list)} samples successfully")
        print(f"Features shape: {features_array.shape}")
        print(f"Labels shape: {labels_array.shape}")
        
        # Class distribution
        print("\nClass distribution:")
        class_names = ['bass', 'brass', 'flute', 'guitar', 'keyboard',
                       'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']
        for i, name in enumerate(class_names):
            count = np.sum(labels_array == i)
            print(f"  {name}: {count} ({count/len(labels_array)*100:.1f}%)")
    
    # Save
    np.savez(
        output_path,
        features=features_array,
        labels=labels_array,
        note_ids=np.array(note_ids_list)
    )
    
    if verbose:
        print(f"\nSaved to {output_path}")
    
    return features_array, labels_array


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Extract Laplace features from NSynth dataset')
    parser.add_argument('--nsynth_dir', type=str, required=True,
                        help='Path to NSynth directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for features (.npz)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples to process')
    parser.add_argument('--n_workers', type=int, default=4,
                        help='Number of parallel workers')
    
    args = parser.parse_args()
    
    extract_nsynth_features(
        nsynth_dir=args.nsynth_dir,
        output_path=args.output,
        max_samples=args.max_samples,
        n_workers=args.n_workers,
        verbose=True
    )


if __name__ == '__main__':
    main()
