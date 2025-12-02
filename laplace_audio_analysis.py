"""
Advanced Audio Analysis: Laplace-inspired methods for music signal processing
Implements three methods:
1. Short-time Prony analysis (explicit frequency + decay rates)
2. Variable-Q wavelet transform (multi-resolution time-frequency)
3. Gammatone filterbank (perceptually-motivated attack/decay analysis)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.linalg import toeplitz, lstsq
import librosa
import librosa.display
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class PronyAnalyzer:
    """Short-time Prony analysis for frequency and damping estimation"""
    
    def __init__(self, n_components: int = 10, model_order: int = 20):
        """
        Args:
            n_components: Number of dominant exponential components to extract
            model_order: Order of the linear prediction model
        """
        self.n_components = n_components
        self.model_order = model_order
    
    def _prony_segment(self, segment: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply Prony's method to a single segment
        
        Returns:
            frequencies: Frequency values in Hz
            damping: Damping coefficients (negative = decay, positive = growth)
            amplitudes: Complex amplitudes
        """
        N = len(segment)
        p = min(self.model_order, N // 2)
        
        # Build Toeplitz matrix for linear prediction
        c = segment[:-1]
        r = segment[1:]
        
        if len(c) < p or len(r) < p:
            return np.array([]), np.array([]), np.array([])
        
        # Construct autocorrelation matrix
        R = toeplitz(c[:p])
        
        try:
            # Solve for LP coefficients
            a, _, _, _ = lstsq(R, r[:p])
            
            # Find roots of characteristic polynomial
            roots = np.roots(np.concatenate([[1], -a]))
            
            # Convert to frequency and damping
            z = np.log(roots) * fs
            frequencies = np.abs(z.imag) / (2 * np.pi)
            damping = z.real
            
            # Estimate amplitudes using least squares
            t = np.arange(N) / fs
            basis = np.array([np.exp(z[k] * t) for k in range(len(z))]).T
            amplitudes, _, _, _ = lstsq(basis, segment)
            
            # Sort by amplitude magnitude and keep top components
            idx = np.argsort(np.abs(amplitudes))[::-1][:self.n_components]
            
            return frequencies[idx], damping[idx], amplitudes[idx]
        
        except:
            return np.array([]), np.array([]), np.array([])
    
    def analyze(self, audio: np.ndarray, sr: float, 
                hop_length: int = 512, win_length: int = 2048) -> dict:
        """
        Perform short-time Prony analysis
        
        Returns:
            dict with 'times', 'frequencies', 'damping', 'amplitudes'
        """
        n_frames = 1 + (len(audio) - win_length) // hop_length
        times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)
        
        all_freqs = []
        all_damping = []
        all_amps = []
        
        window = signal.windows.hann(win_length)
        
        for i in range(n_frames):
            start = i * hop_length
            end = start + win_length
            
            if end > len(audio):
                break
            
            segment = audio[start:end] * window
            freqs, damp, amps = self._prony_segment(segment, sr)
            
            all_freqs.append(freqs)
            all_damping.append(damp)
            all_amps.append(amps)
        
        return {
            'times': times[:len(all_freqs)],
            'frequencies': all_freqs,
            'damping': all_damping,
            'amplitudes': all_amps
        }


class VariableQWavelet:
    """Variable-Q Wavelet Transform for multi-resolution analysis"""
    
    def __init__(self, fmin: float = 55, n_bins: int = 84, 
                 bins_per_octave: int = 12, q_rate: float = 1.0):
        """
        Args:
            fmin: Minimum frequency (Hz)
            n_bins: Number of frequency bins
            bins_per_octave: Frequency resolution
            q_rate: Q factor (higher = better frequency resolution, worse time resolution)
        """
        self.fmin = fmin
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.q_rate = q_rate
    
    def analyze(self, audio: np.ndarray, sr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Variable-Q Transform
        
        Returns:
            vqt: Complex VQT matrix (n_bins x n_frames)
            times: Time coordinates
            frequencies: Frequency coordinates
        """
        # Use librosa's VQT implementation
        vqt = librosa.vqt(audio, sr=sr, 
                         fmin=self.fmin,
                         n_bins=self.n_bins,
                         bins_per_octave=self.bins_per_octave)
        
        # Calculate time and frequency axes
        hop_length = 512
        times = librosa.frames_to_time(np.arange(vqt.shape[1]), sr=sr, hop_length=hop_length)
        frequencies = librosa.cqt_frequencies(self.n_bins, fmin=self.fmin, 
                                             bins_per_octave=self.bins_per_octave)
        
        return vqt, times, frequencies
    
    def get_phase_derivatives(self, vqt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute instantaneous frequency and damping from phase derivatives
        
        Returns:
            inst_freq: Instantaneous frequency deviation
            damping: Estimated damping (from amplitude envelope)
        """
        # Phase derivative over time (instantaneous frequency)
        phase = np.angle(vqt)
        phase_unwrapped = np.unwrap(phase, axis=1)
        inst_freq = np.diff(phase_unwrapped, axis=1)
        
        # Amplitude envelope derivative (related to damping)
        amplitude = np.abs(vqt)
        log_amp = np.log(amplitude + 1e-10)
        damping = np.diff(log_amp, axis=1)
        
        return inst_freq, damping


class GammatoneFilterbank:
    """Gammatone filterbank for perceptually-motivated analysis"""
    
    def __init__(self, n_filters: int = 64, fmin: float = 50, fmax: float = 8000):
        """
        Args:
            n_filters: Number of gammatone filters
            fmin: Minimum center frequency
            fmax: Maximum center frequency
        """
        self.n_filters = n_filters
        self.fmin = fmin
        self.fmax = fmax
    
    def _erb_space(self) -> np.ndarray:
        """Generate ERB-spaced center frequencies"""
        # ERB (Equivalent Rectangular Bandwidth) scale
        erb_min = self._freq_to_erb(self.fmin)
        erb_max = self._freq_to_erb(self.fmax)
        erb_points = np.linspace(erb_min, erb_max, self.n_filters)
        return self._erb_to_freq(erb_points)
    
    def _freq_to_erb(self, freq: float) -> float:
        """Convert frequency to ERB scale"""
        return 9.265 * np.log(1 + freq / (24.7 * 9.265))
    
    def _erb_to_freq(self, erb: np.ndarray) -> np.ndarray:
        """Convert ERB scale to frequency"""
        return 24.7 * 9.265 * (np.exp(erb / 9.265) - 1)
    
    def _gammatone_impulse(self, cf: float, sr: float, duration: float = 0.1) -> np.ndarray:
        """
        Generate gammatone impulse response
        
        Args:
            cf: Center frequency
            sr: Sample rate
            duration: Duration in seconds
        """
        n = 4  # Filter order
        erb = 24.7 * (4.37 * cf / 1000 + 1)  # ERB bandwidth
        b = 1.019 * erb
        
        t = np.arange(0, duration, 1/sr)
        
        # Gammatone impulse response
        env = (t ** (n - 1)) * np.exp(-2 * np.pi * b * t)
        carrier = np.cos(2 * np.pi * cf * t)
        
        return env * carrier
    
    def analyze(self, audio: np.ndarray, sr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply gammatone filterbank
        
        Returns:
            filtered: Filtered signals (n_filters x n_samples)
            envelopes: Hilbert envelopes (attack/decay characteristics)
            center_freqs: Center frequencies of filters
        """
        center_freqs = self._erb_space()
        n_samples = len(audio)
        
        filtered = np.zeros((self.n_filters, n_samples))
        envelopes = np.zeros((self.n_filters, n_samples))
        
        for i, cf in enumerate(center_freqs):
            # Create gammatone filter
            impulse = self._gammatone_impulse(cf, sr)
            
            # Apply filter
            filtered[i] = signal.convolve(audio, impulse, mode='same')
            
            # Extract envelope using Hilbert transform
            analytic = signal.hilbert(filtered[i])
            envelopes[i] = np.abs(analytic)
        
        return filtered, envelopes, center_freqs


def generate_test_signal(sr: float = 22050, duration: float = 2.0) -> np.ndarray:
    """Generate a test signal with multiple decaying sinusoids (like piano notes)"""
    t = np.arange(0, duration, 1/sr)
    
    # Three notes with different decay rates
    notes = [
        {'freq': 440, 'decay': 2.0, 'amp': 1.0},    # A4 - slow decay
        {'freq': 554.37, 'decay': 4.0, 'amp': 0.8},  # C#5 - medium decay
        {'freq': 659.25, 'decay': 6.0, 'amp': 0.6},  # E5 - fast decay
    ]
    
    signal_out = np.zeros_like(t)
    
    for note in notes:
        envelope = note['amp'] * np.exp(-note['decay'] * t)
        tone = envelope * np.sin(2 * np.pi * note['freq'] * t)
        signal_out += tone
    
    # Add some noise
    signal_out += 0.01 * np.random.randn(len(t))
    
    return signal_out


def visualize_prony(prony_results: dict, sr: float):
    """Visualize Prony analysis results"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    times = prony_results['times']
    
    # Plot 1: Frequency tracks
    ax = axes[0]
    for i, (freqs, amps) in enumerate(zip(prony_results['frequencies'], 
                                          prony_results['amplitudes'])):
        if len(freqs) > 0:
            # Size by amplitude
            sizes = np.abs(amps) * 100
            ax.scatter([times[i]] * len(freqs), freqs, s=sizes, alpha=0.6)
    
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Short-time Prony Analysis: Frequency Tracks')
    ax.set_ylim([0, 2000])
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Damping coefficients
    ax = axes[1]
    for i, (freqs, damp, amps) in enumerate(zip(prony_results['frequencies'],
                                                 prony_results['damping'],
                                                 prony_results['amplitudes'])):
        if len(damp) > 0:
            sizes = np.abs(amps) * 100
            colors = ['red' if d < 0 else 'blue' for d in damp]
            ax.scatter([times[i]] * len(damp), damp, s=sizes, c=colors, alpha=0.6)
    
    ax.set_ylabel('Damping Coefficient')
    ax.set_title('Damping Rates (Red=Decay, Blue=Growth)')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Frequency vs Damping (phase space)
    ax = axes[2]
    all_freqs = np.concatenate([f for f in prony_results['frequencies'] if len(f) > 0])
    all_damp = np.concatenate([d for d in prony_results['damping'] if len(d) > 0])
    all_amps = np.concatenate([a for a in prony_results['amplitudes'] if len(a) > 0])
    
    if len(all_freqs) > 0:
        sizes = np.abs(all_amps) * 50
        ax.scatter(all_freqs, all_damp, s=sizes, alpha=0.5, c=all_freqs, cmap='viridis')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Damping Coefficient')
        ax.set_title('Frequency-Damping Phase Space')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 2000])
    
    plt.tight_layout()
    return fig


def visualize_vqt(vqt: np.ndarray, times: np.ndarray, frequencies: np.ndarray,
                  inst_freq: np.ndarray, damping: np.ndarray):
    """Visualize Variable-Q Transform results"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: VQT magnitude spectrogram
    ax = axes[0]
    vqt_db = librosa.amplitude_to_db(np.abs(vqt), ref=np.max)
    img = ax.pcolormesh(times, frequencies, vqt_db, shading='auto', cmap='magma')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Variable-Q Transform (VQT) - Magnitude')
    ax.set_yscale('log')
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    
    # Plot 2: Instantaneous frequency deviation
    ax = axes[1]
    img = ax.pcolormesh(times[:-1], frequencies, inst_freq, 
                       shading='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Instantaneous Frequency Deviation (Phase Derivative)')
    ax.set_yscale('log')
    plt.colorbar(img, ax=ax, label='Phase change')
    
    # Plot 3: Damping estimate (amplitude envelope derivative)
    ax = axes[2]
    img = ax.pcolormesh(times[:-1], frequencies, damping,
                       shading='auto', cmap='RdYlGn_r', vmin=-0.1, vmax=0.1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Estimated Damping (Amplitude Envelope Slope)')
    ax.set_yscale('log')
    plt.colorbar(img, ax=ax, label='Log amplitude change')
    
    plt.tight_layout()
    return fig


def visualize_gammatone(filtered: np.ndarray, envelopes: np.ndarray,
                       center_freqs: np.ndarray, sr: float, duration: float = 0.5):
    """Visualize Gammatone filterbank results"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Subsample for visualization
    n_samples_plot = int(duration * sr)
    times = np.arange(n_samples_plot) / sr
    
    # Plot 1: Filterbank output (first few filters)
    ax = axes[0]
    n_display = min(10, filtered.shape[0])
    for i in range(n_display):
        ax.plot(times, filtered[i, :n_samples_plot] + i*2, 
               alpha=0.7, linewidth=0.5)
    ax.set_ylabel('Filter Output (offset)')
    ax.set_title(f'Gammatone Filterbank Outputs (first {n_display} filters)')
    ax.set_xlim([0, duration])
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Envelopes spectrogram
    ax = axes[1]
    # Downsample envelopes for visualization
    hop = 256
    env_ds = envelopes[:, ::hop]
    times_ds = np.arange(env_ds.shape[1]) * hop / sr
    
    img = ax.pcolormesh(times_ds, center_freqs, 
                       librosa.amplitude_to_db(env_ds + 1e-10, ref=np.max),
                       shading='auto', cmap='hot')
    ax.set_ylabel('Center Frequency (Hz)')
    ax.set_title('Gammatone Envelope Spectrogram')
    ax.set_yscale('log')
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    
    # Plot 3: Envelope attack/decay analysis
    ax = axes[2]
    # Compute envelope derivatives (attack/decay rate)
    env_deriv = np.diff(np.log(envelopes[:, ::hop] + 1e-10), axis=1)
    
    img = ax.pcolormesh(times_ds[:-1], center_freqs, env_deriv,
                       shading='auto', cmap='RdBu_r', vmin=-0.01, vmax=0.01)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Center Frequency (Hz)')
    ax.set_title('Attack/Decay Characteristics (Envelope Derivative)')
    ax.set_yscale('log')
    plt.colorbar(img, ax=ax, label='Envelope slope')
    
    plt.tight_layout()
    return fig


def main():
    """Main execution function"""
    print("=" * 70)
    print("Advanced Audio Analysis: Laplace-inspired Methods")
    print("=" * 70)
    
    # Generate or load test signal
    sr = 22050
    print(f"\n1. Generating test signal (3 decaying notes)...")
    audio = generate_test_signal(sr=sr, duration=2.0)
    
    # Alternatively, load a real audio file:
    # audio, sr = librosa.load('your_audio.wav', sr=22050, duration=5)
    
    print(f"   Sample rate: {sr} Hz")
    print(f"   Duration: {len(audio)/sr:.2f} seconds")
    
    # Method 1: Short-time Prony Analysis
    print("\n2. Performing Short-time Prony Analysis...")
    prony = PronyAnalyzer(n_components=5, model_order=30)
    prony_results = prony.analyze(audio, sr, hop_length=512, win_length=2048)
    print(f"   Analyzed {len(prony_results['times'])} time frames")
    
    fig1 = visualize_prony(prony_results, sr)
    plt.savefig('/mnt/user-data/outputs/prony_analysis.png', dpi=150, bbox_inches='tight')
    print("   ✓ Visualization saved: prony_analysis.png")
    
    # Method 2: Variable-Q Wavelet Transform
    print("\n3. Computing Variable-Q Wavelet Transform...")
    vqwt = VariableQWavelet(fmin=55, n_bins=84, bins_per_octave=12)
    vqt, times_vqt, freqs_vqt = vqwt.analyze(audio, sr)
    inst_freq, damping = vqwt.get_phase_derivatives(vqt)
    print(f"   VQT shape: {vqt.shape}")
    print(f"   Frequency range: {freqs_vqt[0]:.1f} - {freqs_vqt[-1]:.1f} Hz")
    
    fig2 = visualize_vqt(vqt, times_vqt, freqs_vqt, inst_freq, damping)
    plt.savefig('/mnt/user-data/outputs/vqt_analysis.png', dpi=150, bbox_inches='tight')
    print("   ✓ Visualization saved: vqt_analysis.png")
    
    # Method 3: Gammatone Filterbank
    print("\n4. Applying Gammatone Filterbank...")
    gtfb = GammatoneFilterbank(n_filters=64, fmin=50, fmax=8000)
    filtered, envelopes, center_freqs = gtfb.analyze(audio, sr)
    print(f"   Number of filters: {len(center_freqs)}")
    print(f"   Frequency range: {center_freqs[0]:.1f} - {center_freqs[-1]:.1f} Hz")
    
    fig3 = visualize_gammatone(filtered, envelopes, center_freqs, sr, duration=1.0)
    plt.savefig('/mnt/user-data/outputs/gammatone_analysis.png', dpi=150, bbox_inches='tight')
    print("   ✓ Visualization saved: gammatone_analysis.png")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
    print("\nKey insights from each method:")
    print("\n• Prony Analysis:")
    print("  - Extracts explicit frequency + damping pairs")
    print("  - Red dots = decaying components (negative damping)")
    print("  - Blue dots = growing components (usually artifacts)")
    print("  - Dot size = amplitude")
    
    print("\n• Variable-Q Wavelet:")
    print("  - Multi-resolution time-frequency representation")
    print("  - Phase derivative shows frequency modulation")
    print("  - Amplitude slope reveals attack/decay behavior")
    print("  - Logarithmic frequency scale = better for music")
    
    print("\n• Gammatone Filterbank:")
    print("  - Perceptually-motivated (mimics human auditory system)")
    print("  - Envelope tracking captures attack/decay naturally")
    print("  - Red = attack (increasing amplitude)")
    print("  - Blue = decay (decreasing amplitude)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
