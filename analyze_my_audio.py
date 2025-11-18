"""
Simple Example: Analyze Your Own Audio File
============================================

This script shows how to use the three Laplace-inspired methods
on your own audio file with minimal code.

Usage:
    python analyze_my_audio.py path/to/your/audio.wav
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
from laplace_audio_analysis import (
    PronyAnalyzer, 
    VariableQWavelet, 
    GammatoneFilterbank,
    visualize_prony,
    visualize_vqt,
    visualize_gammatone
)


def analyze_audio_file(audio_path, duration=10.0, method='all'):
    """
    Analyze an audio file with Laplace-inspired methods
    
    Args:
        audio_path: Path to audio file (wav, mp3, flac, etc.)
        duration: Duration in seconds to analyze (None = full file)
        method: Which method to use ('prony', 'vqt', 'gammatone', or 'all')
    """
    
    print(f"\n{'='*70}")
    print(f"Analyzing: {audio_path}")
    print(f"{'='*70}\n")
    
    # Load audio file
    print("Loading audio file...")
    audio, sr = librosa.load(audio_path, sr=22050, duration=duration)
    print(f"✓ Loaded: {len(audio)/sr:.2f} seconds at {sr} Hz")
    print(f"  Shape: {audio.shape}")
    print(f"  Range: [{audio.min():.3f}, {audio.max():.3f}]")
    
    results = {}
    
    # Method 1: Prony Analysis
    if method in ['prony', 'all']:
        print("\n1. Running Prony Analysis...")
        prony = PronyAnalyzer(n_components=10, model_order=30)
        prony_results = prony.analyze(audio, sr, hop_length=512, win_length=2048)
        results['prony'] = prony_results
        
        print(f"   ✓ Analyzed {len(prony_results['times'])} frames")
        
        # Count total components detected
        total_components = sum(len(f) for f in prony_results['frequencies'])
        print(f"   ✓ Detected {total_components} frequency components")
        
        # Visualize
        fig = visualize_prony(prony_results, sr)
        output_path = audio_path.rsplit('.', 1)[0] + '_prony.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: {output_path}")
    
    # Method 2: Variable-Q Wavelet
    if method in ['vqt', 'all']:
        print("\n2. Running Variable-Q Wavelet Transform...")
        vqwt = VariableQWavelet(fmin=55, n_bins=84, bins_per_octave=12)
        vqt, times, freqs = vqwt.analyze(audio, sr)
        inst_freq, damping = vqwt.get_phase_derivatives(vqt)
        results['vqt'] = {
            'vqt': vqt,
            'times': times,
            'frequencies': freqs,
            'inst_freq': inst_freq,
            'damping': damping
        }
        
        print(f"   ✓ VQT shape: {vqt.shape}")
        print(f"   ✓ Frequency range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz")
        
        # Visualize
        fig = visualize_vqt(vqt, times, freqs, inst_freq, damping)
        output_path = audio_path.rsplit('.', 1)[0] + '_vqt.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: {output_path}")
    
    # Method 3: Gammatone Filterbank
    if method in ['gammatone', 'all']:
        print("\n3. Running Gammatone Filterbank...")
        gtfb = GammatoneFilterbank(n_filters=64, fmin=50, fmax=8000)
        filtered, envelopes, center_freqs = gtfb.analyze(audio, sr)
        results['gammatone'] = {
            'filtered': filtered,
            'envelopes': envelopes,
            'center_freqs': center_freqs
        }
        
        print(f"   ✓ Applied {len(center_freqs)} filters")
        print(f"   ✓ Frequency range: {center_freqs[0]:.1f} - {center_freqs[-1]:.1f} Hz")
        
        # Visualize
        fig = visualize_gammatone(filtered, envelopes, center_freqs, sr, duration=2.0)
        output_path = audio_path.rsplit('.', 1)[0] + '_gammatone.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved: {output_path}")
    
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}\n")
    
    return results


def quick_comparison(audio_path, duration=5.0):
    """
    Quick comparison showing all three methods side-by-side
    """
    print(f"\nQuick Comparison Mode")
    print(f"Analyzing first {duration} seconds...\n")
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=22050, duration=duration)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 1. Prony - show frequency tracks
    prony = PronyAnalyzer(n_components=8, model_order=25)
    prony_results = prony.analyze(audio, sr, hop_length=512, win_length=2048)
    
    ax = axes[0]
    for i, (freqs, amps) in enumerate(zip(prony_results['frequencies'], 
                                          prony_results['amplitudes'])):
        if len(freqs) > 0:
            times = [prony_results['times'][i]] * len(freqs)
            sizes = np.abs(amps) * 100
            ax.scatter(times, freqs, s=sizes, alpha=0.6, c='red')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Method 1: Prony Analysis (Explicit Frequency + Damping)')
    ax.set_ylim([0, 2000])
    ax.grid(True, alpha=0.3)
    
    # 2. VQT - show spectrogram
    vqwt = VariableQWavelet(fmin=55, n_bins=72, bins_per_octave=12)
    vqt, times, freqs = vqwt.analyze(audio, sr)
    
    ax = axes[1]
    vqt_db = librosa.amplitude_to_db(np.abs(vqt), ref=np.max)
    img = ax.pcolormesh(times, freqs, vqt_db, shading='auto', cmap='magma')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Method 2: Variable-Q Wavelet (Multi-resolution Time-Frequency)')
    ax.set_yscale('log')
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    
    # 3. Gammatone - show envelope spectrogram
    gtfb = GammatoneFilterbank(n_filters=48, fmin=50, fmax=4000)
    filtered, envelopes, center_freqs = gtfb.analyze(audio, sr)
    
    ax = axes[2]
    hop = 256
    env_ds = envelopes[:, ::hop]
    times_ds = np.arange(env_ds.shape[1]) * hop / sr
    img = ax.pcolormesh(times_ds, center_freqs, 
                       librosa.amplitude_to_db(env_ds + 1e-10, ref=np.max),
                       shading='auto', cmap='hot')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Method 3: Gammatone Filterbank (Perceptual Envelope Analysis)')
    ax.set_yscale('log')
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    
    plt.tight_layout()
    
    # Save
    output_path = audio_path.rsplit('.', 1)[0] + '_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved comparison: {output_path}\n")


def print_usage():
    """Print usage instructions"""
    print("""
Usage:
    python analyze_my_audio.py <audio_file> [options]

Examples:
    # Analyze with all methods
    python analyze_my_audio.py my_song.wav
    
    # Quick comparison (first 5 seconds)
    python analyze_my_audio.py my_song.wav --quick
    
    # Use only one method
    python analyze_my_audio.py my_song.wav --method prony
    python analyze_my_audio.py my_song.wav --method vqt
    python analyze_my_audio.py my_song.wav --method gammatone
    
    # Analyze specific duration
    python analyze_my_audio.py my_song.wav --duration 30
    
Options:
    --method <name>      Which method: prony, vqt, gammatone, or all (default: all)
    --duration <sec>     Duration in seconds to analyze (default: 10.0)
    --quick              Quick comparison mode (3 methods side-by-side)

Supported formats: wav, mp3, flac, ogg, m4a (anything librosa supports)
""")


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help']:
        print_usage()
        sys.exit(0)
    
    audio_path = sys.argv[1]
    
    # Parse options
    method = 'all'
    duration = 10.0
    quick_mode = False
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--method' and i + 1 < len(sys.argv):
            method = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--duration' and i + 1 < len(sys.argv):
            duration = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--quick':
            quick_mode = True
            i += 1
        else:
            print(f"Unknown option: {sys.argv[i]}")
            print_usage()
            sys.exit(1)
    
    # Check if file exists
    import os
    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)
    
    # Run analysis
    try:
        if quick_mode:
            quick_comparison(audio_path, duration=min(duration, 5.0))
        else:
            analyze_audio_file(audio_path, duration=duration, method=method)
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
