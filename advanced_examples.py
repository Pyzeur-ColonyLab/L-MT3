"""
Advanced Usage Examples for Polyphonic Transcription
Practical applications for audio-to-MIDI with decay information
"""

import numpy as np
import librosa
from laplace_audio_analysis import PronyAnalyzer, VariableQWavelet, GammatoneFilterbank
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt


def separate_notes_by_decay(audio: np.ndarray, sr: float) -> dict:
    """
    Use Prony analysis to separate overlapping notes by their decay characteristics
    
    Example: Piano chord where notes have different damping rates
    """
    prony = PronyAnalyzer(n_components=15, model_order=40)
    results = prony.analyze(audio, sr, hop_length=256, win_length=2048)
    
    # Collect all detected components
    all_components = []
    for i, (freqs, damps, amps) in enumerate(zip(
        results['frequencies'], 
        results['damping'], 
        results['amplitudes']
    )):
        for f, d, a in zip(freqs, damps, amps):
            if f > 50 and f < 2000 and d < 0:  # Valid musical range, decaying
                all_components.append({
                    'frequency': f,
                    'damping': d,
                    'amplitude': abs(a),
                    'time': results['times'][i]
                })
    
    if len(all_components) == 0:
        return {'notes': [], 'clusters': []}
    
    # Cluster by frequency (group into notes)
    freqs = np.array([c['frequency'] for c in all_components])
    freq_clusters = []
    
    # Simple frequency binning (12-TET)
    for f in freqs:
        midi = 69 + 12 * np.log2(f / 440)
        midi_rounded = round(midi)
        freq_clusters.append(midi_rounded)
    
    # Group components by MIDI note
    notes = {}
    for comp, midi in zip(all_components, freq_clusters):
        if midi not in notes:
            notes[midi] = []
        notes[midi].append(comp)
    
    # Analyze decay statistics per note
    note_info = []
    for midi, components in notes.items():
        damps = [c['damping'] for c in components]
        amps = [c['amplitude'] for c in components]
        
        note_info.append({
            'midi': midi,
            'frequency': 440 * 2**((midi - 69) / 12),
            'mean_damping': np.mean(damps),
            'std_damping': np.std(damps),
            'mean_amplitude': np.mean(amps),
            'n_detections': len(components)
        })
    
    return {
        'notes': note_info,
        'raw_components': all_components
    }


def detect_onsets_with_gammatone(audio: np.ndarray, sr: float, 
                                 threshold: float = 0.3) -> dict:
    """
    Use Gammatone filterbank for robust onset detection
    
    Returns onset times with frequency information
    """
    gtfb = GammatoneFilterbank(n_filters=48, fmin=80, fmax=4000)
    filtered, envelopes, center_freqs = gtfb.analyze(audio, sr)
    
    # Compute envelope derivatives (onset strength)
    hop = 512
    envelopes_ds = envelopes[:, ::hop]
    times = np.arange(envelopes_ds.shape[1]) * hop / sr
    
    onset_strength = np.diff(envelopes_ds, axis=1)
    onset_strength = np.maximum(onset_strength, 0)  # Only positive changes
    
    # Detect peaks in each frequency band
    onsets = []
    for i, cf in enumerate(center_freqs):
        # Simple peak detection
        strength = onset_strength[i]
        threshold_abs = threshold * np.max(strength)
        
        for t_idx in range(1, len(strength) - 1):
            if (strength[t_idx] > threshold_abs and 
                strength[t_idx] > strength[t_idx - 1] and
                strength[t_idx] > strength[t_idx + 1]):
                
                onsets.append({
                    'time': times[t_idx],
                    'frequency': cf,
                    'strength': strength[t_idx]
                })
    
    # Sort by time
    onsets = sorted(onsets, key=lambda x: x['time'])
    
    return {
        'onsets': onsets,
        'envelope_matrix': envelopes_ds,
        'times': times,
        'frequencies': center_freqs
    }


def vqt_pitch_tracking(audio: np.ndarray, sr: float) -> dict:
    """
    Use VQT for polyphonic pitch tracking with temporal coherence
    
    Tracks multiple simultaneous pitches over time
    """
    vqwt = VariableQWavelet(fmin=65, n_bins=72, bins_per_octave=12)
    vqt, times, freqs = vqwt.analyze(audio, sr)
    
    # Get magnitude
    vqt_mag = np.abs(vqt)
    
    # Peak picking in each time frame
    pitch_tracks = []
    
    for t_idx in range(vqt_mag.shape[1]):
        frame = vqt_mag[:, t_idx]
        
        # Find local maxima
        peaks = []
        for f_idx in range(1, len(frame) - 1):
            if (frame[f_idx] > frame[f_idx - 1] and 
                frame[f_idx] > frame[f_idx + 1] and
                frame[f_idx] > 0.1 * np.max(frame)):
                
                peaks.append({
                    'time': times[t_idx],
                    'frequency': freqs[f_idx],
                    'magnitude': frame[f_idx],
                    'bin': f_idx
                })
        
        pitch_tracks.append(peaks)
    
    # Link peaks across time (simple nearest-neighbor tracking)
    trajectories = []
    used = set()
    
    for t_idx in range(len(pitch_tracks) - 1):
        for peak in pitch_tracks[t_idx]:
            if (t_idx, peak['bin']) in used:
                continue
                
            # Find continuation in next frame
            trajectory = [peak]
            current_bin = peak['bin']
            
            for future_t in range(t_idx + 1, min(t_idx + 10, len(pitch_tracks))):
                # Look for nearby frequency
                candidates = [p for p in pitch_tracks[future_t] 
                            if abs(p['bin'] - current_bin) <= 2]
                
                if not candidates:
                    break
                
                # Take strongest
                next_peak = max(candidates, key=lambda x: x['magnitude'])
                trajectory.append(next_peak)
                current_bin = next_peak['bin']
                used.add((future_t, current_bin))
            
            if len(trajectory) >= 3:  # Minimum trajectory length
                trajectories.append(trajectory)
    
    return {
        'trajectories': trajectories,
        'vqt_magnitude': vqt_mag,
        'times': times,
        'frequencies': freqs
    }


def combine_methods_for_transcription(audio: np.ndarray, sr: float) -> dict:
    """
    Comprehensive analysis combining all three methods
    
    1. Gammatone: Detect onsets
    2. VQT: Track pitches over time
    3. Prony: Estimate decay rates per note
    """
    print("Running comprehensive analysis...")
    
    # Step 1: Onset detection
    print("  1/3 Detecting onsets with Gammatone...")
    onset_results = detect_onsets_with_gammatone(audio, sr, threshold=0.3)
    onsets = onset_results['onsets']
    print(f"      Found {len(onsets)} onsets")
    
    # Step 2: Pitch tracking
    print("  2/3 Tracking pitches with VQT...")
    pitch_results = vqt_pitch_tracking(audio, sr)
    trajectories = pitch_results['trajectories']
    print(f"      Found {len(trajectories)} pitch trajectories")
    
    # Step 3: Decay analysis
    print("  3/3 Analyzing decay with Prony...")
    decay_results = separate_notes_by_decay(audio, sr)
    notes = decay_results['notes']
    print(f"      Identified {len(notes)} distinct notes")
    
    # Combine information
    combined_notes = []
    
    for traj in trajectories:
        start_time = traj[0]['time']
        end_time = traj[-1]['time']
        mean_freq = np.mean([p['frequency'] for p in traj])
        midi = 69 + 12 * np.log2(mean_freq / 440)
        
        # Find matching Prony components
        matching_note = None
        for note in notes:
            if abs(note['midi'] - round(midi)) <= 1:
                matching_note = note
                break
        
        # Find onset
        onset_time = None
        for onset in onsets:
            if (abs(onset['time'] - start_time) < 0.1 and 
                abs(onset['frequency'] - mean_freq) < 100):
                onset_time = onset['time']
                break
        
        combined_notes.append({
            'onset': onset_time if onset_time else start_time,
            'offset': end_time,
            'midi': round(midi),
            'frequency': mean_freq,
            'damping': matching_note['mean_damping'] if matching_note else None,
            'confidence': len(traj) / 10.0  # Trajectory length as confidence
        })
    
    return {
        'notes': combined_notes,
        'onsets': onsets,
        'trajectories': trajectories,
        'decay_analysis': notes
    }


def visualize_combined_analysis(results: dict, audio: np.ndarray, sr: float):
    """Visualize the combined analysis results"""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    duration = len(audio) / sr
    times_audio = np.linspace(0, duration, len(audio))
    
    # Plot 1: Waveform with detected onsets
    ax = axes[0]
    ax.plot(times_audio, audio, alpha=0.5, linewidth=0.5)
    for onset in results['onsets'][:50]:  # Limit displayed onsets
        ax.axvline(onset['time'], color='red', alpha=0.3, linewidth=1)
    ax.set_ylabel('Amplitude')
    ax.set_title('Waveform with Detected Onsets (Gammatone)')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Pitch trajectories
    ax = axes[1]
    for traj in results['trajectories']:
        times = [p['time'] for p in traj]
        freqs = [p['frequency'] for p in traj]
        ax.plot(times, freqs, marker='o', markersize=2, alpha=0.7)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Pitch Trajectories (VQT)')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Note events (piano roll style)
    ax = axes[2]
    for note in results['notes']:
        midi = note['midi']
        onset = note['onset']
        offset = note['offset']
        duration = offset - onset
        
        # Color by damping if available
        if note['damping'] is not None:
            color = plt.cm.RdYlGn(-note['damping'] * 10)
        else:
            color = 'gray'
        
        ax.barh(midi, duration, left=onset, height=0.8, 
               color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('MIDI Note')
    ax.set_title('Detected Notes (Combined Analysis) - Color = Decay Rate')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Decay characteristics
    ax = axes[3]
    decay_notes = results['decay_analysis']
    if len(decay_notes) > 0:
        midis = [n['midi'] for n in decay_notes]
        damps = [n['mean_damping'] for n in decay_notes]
        amps = [n['mean_amplitude'] for n in decay_notes]
        
        scatter = ax.scatter(midis, damps, s=np.array(amps)*500, 
                           alpha=0.6, c=damps, cmap='RdYlGn_r')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('MIDI Note')
        ax.set_ylabel('Damping Coefficient')
        ax.set_title('Decay Characteristics per Note (Prony)')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Damping')
    
    plt.tight_layout()
    return fig


def example_usage():
    """Example demonstrating all advanced features"""
    
    # Generate a more complex test signal
    sr = 22050
    duration = 3.0
    t = np.arange(0, duration, 1/sr)
    
    # Simulated piano chord with staggered onsets and different decays
    notes = [
        {'midi': 60, 'onset': 0.0, 'decay': 2.0},   # C4
        {'midi': 64, 'onset': 0.05, 'decay': 3.0},  # E4
        {'midi': 67, 'onset': 0.1, 'decay': 4.0},   # G4
        {'midi': 72, 'onset': 0.5, 'decay': 5.0},   # C5
    ]
    
    audio = np.zeros_like(t)
    for note in notes:
        freq = 440 * 2**((note['midi'] - 69) / 12)
        onset_idx = int(note['onset'] * sr)
        
        t_note = t[onset_idx:] - note['onset']
        envelope = np.exp(-note['decay'] * t_note)
        tone = envelope * np.sin(2 * np.pi * freq * t_note)
        
        # Add harmonics
        tone += 0.3 * envelope * np.sin(2 * np.pi * freq * 2 * t_note)
        tone += 0.15 * envelope * np.sin(2 * np.pi * freq * 3 * t_note)
        
        audio[onset_idx:] += tone
    
    # Add noise
    audio += 0.02 * np.random.randn(len(audio))
    audio = audio / np.max(np.abs(audio))
    
    print("\n" + "="*70)
    print("Advanced Polyphonic Transcription Demo")
    print("="*70)
    
    # Run comprehensive analysis
    results = combine_methods_for_transcription(audio, sr)
    
    print("\n" + "="*70)
    print("Results Summary:")
    print("="*70)
    print(f"\nDetected {len(results['notes'])} notes:")
    for i, note in enumerate(results['notes'], 1):
        print(f"\n  Note {i}:")
        print(f"    MIDI: {note['midi']}")
        print(f"    Frequency: {note['frequency']:.1f} Hz")
        print(f"    Onset: {note['onset']:.3f} s")
        print(f"    Offset: {note['offset']:.3f} s")
        print(f"    Duration: {note['offset'] - note['onset']:.3f} s")
        if note['damping']:
            print(f"    Damping: {note['damping']:.3f}")
        print(f"    Confidence: {note['confidence']:.2f}")
    
    # Visualize
    print("\nGenerating visualization...")
    fig = visualize_combined_analysis(results, audio, sr)
    plt.savefig('/mnt/user-data/outputs/advanced_transcription_demo.png', 
                dpi=150, bbox_inches='tight')
    print("✓ Saved: advanced_transcription_demo.png")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    example_usage()
