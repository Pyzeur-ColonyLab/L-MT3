# Laplace-Inspired Audio Analysis for Music Processing

## Overview

This implementation demonstrates three advanced methods for audio analysis that capture both **frequency** and **temporal decay characteristics** - going beyond standard STFT to extract Laplace-like information from music signals.

## Why These Methods?

Traditional STFT assumes eternal sinusoids. Music signals have:
- **Attack/Decay envelopes** (piano vs synth differ in their decay)
- **Harmonic interactions** (simultaneous notes with different damping rates)
- **Transient behavior** (percussive vs sustained notes)

These three methods capture this richer temporal structure.

---

## Method 1: Short-Time Prony Analysis

### What It Does
Decomposes each audio segment into a sum of **exponentially damped sinusoids**:
```
x(t) = Σ Aₖ e^((σₖ + j2πfₖ)t)
```
where:
- `fₖ` = frequency (Hz)
- `σₖ` = damping coefficient (negative = decay, positive = growth)
- `Aₖ` = complex amplitude

### Key Features
- ✅ **Explicit frequency + damping pairs** for each component
- ✅ Directly related to Laplace pole-residue decomposition
- ✅ Can distinguish notes by their decay characteristics
- ⚠️ Sensitive to noise and model order selection

### Visualization Output
- **Top panel**: Frequency tracks over time (dot size = amplitude)
- **Middle panel**: Damping coefficients (red = decay, blue = growth)
- **Bottom panel**: Frequency-Damping phase space

### Use Cases
- Distinguishing simultaneous notes with different decay rates
- Modeling resonant systems (guitar body, room acoustics)
- Audio-to-MIDI with envelope information

---

## Method 2: Variable-Q Wavelet Transform (VQT)

### What It Does
Multi-resolution time-frequency analysis with:
- **Logarithmic frequency spacing** (matches musical scales)
- **Adaptive time-frequency resolution** (high Q = better frequency resolution)
- **Phase and amplitude tracking** for instantaneous characteristics

### Key Features
- ✅ Captures multi-scale temporal structure
- ✅ Perceptually-aligned frequency resolution
- ✅ Phase derivatives reveal frequency modulation and damping
- ✅ Better than CQT for capturing transients

### Visualization Output
- **Top panel**: VQT magnitude spectrogram (log frequency scale)
- **Middle panel**: Instantaneous frequency deviation (phase derivative)
- **Bottom panel**: Estimated damping (amplitude envelope slope)

### Use Cases
- Polyphonic music transcription
- Music structure analysis
- Pitch tracking with vibrato

---

## Method 3: Gammatone Filterbank

### What It Does
Mimics the **human auditory system** with:
- **ERB-spaced filters** (Equivalent Rectangular Bandwidth)
- **Asymmetric impulse responses** (realistic attack/decay)
- **Envelope extraction** via Hilbert transform

### Key Features
- ✅ Perceptually-motivated (how humans actually hear)
- ✅ Natural attack/decay extraction
- ✅ Works well for onset detection
- ✅ Robust to noise

### Visualization Output
- **Top panel**: Individual filter outputs (first 10 filters)
- **Middle panel**: Envelope spectrogram across all filters
- **Bottom panel**: Attack/Decay characteristics (envelope derivative)

### Use Cases
- Onset detection and rhythm analysis
- Timbre analysis
- Auditory scene analysis (source separation)

---

## Installation

```bash
pip install -r requirements.txt
```

Dependencies:
- numpy (numerical computation)
- scipy (signal processing)
- matplotlib (visualization)
- librosa (audio processing)
- soundfile (audio I/O)

---

## Usage

### Basic Usage (Test Signal)
```bash
python laplace_audio_analysis.py
```
Generates a synthetic 3-note chord with different decay rates.

### With Your Own Audio
```python
import librosa
from laplace_audio_analysis import PronyAnalyzer, VariableQWavelet, GammatoneFilterbank

# Load your audio
audio, sr = librosa.load('your_track.wav', sr=22050, duration=5)

# Method 1: Prony Analysis
prony = PronyAnalyzer(n_components=10, model_order=30)
results = prony.analyze(audio, sr, hop_length=512, win_length=2048)

# Method 2: Variable-Q Wavelet
vqwt = VariableQWavelet(fmin=55, n_bins=84, bins_per_octave=12)
vqt, times, freqs = vqwt.analyze(audio, sr)
inst_freq, damping = vqwt.get_phase_derivatives(vqt)

# Method 3: Gammatone Filterbank
gtfb = GammatoneFilterbank(n_filters=64, fmin=50, fmax=8000)
filtered, envelopes, center_freqs = gtfb.analyze(audio, sr)
```

---

## Parameter Tuning

### Prony Analysis
- `n_components`: Number of dominant components to extract (5-20 typical)
- `model_order`: Linear prediction order (20-50 typical)
- Higher order = more components but more noise sensitivity

### Variable-Q Wavelet
- `fmin`: Lowest frequency (typically 20-100 Hz)
- `bins_per_octave`: Frequency resolution (12 = semitones, 36 = third-tones)
- `q_rate`: Quality factor (1.0 standard, higher = better freq resolution)

### Gammatone Filterbank
- `n_filters`: Number of filters (32-128 typical)
- `fmin/fmax`: Frequency range of interest
- ERB spacing is automatic based on psychoacoustics

---

## Comparison to Standard Methods

| Method | Time Res | Freq Res | Damping Info | Computational Cost | Best For |
|--------|----------|----------|--------------|-------------------|----------|
| **STFT** | Fixed | Fixed | ❌ None | Low | General spectral analysis |
| **Prony** | Segment | Explicit | ✅ Explicit σ | Medium | Modeling resonant systems |
| **VQT** | Adaptive | Log-scale | ⚠️ From phase | Medium-High | Polyphonic music |
| **Gammatone** | Adaptive | ERB-scale | ✅ From envelope | Medium | Perceptual modeling |

---

## Applications for Audio-to-MIDI

For polyphonic transcription (like MT3 improvements):

1. **Prony Analysis** → Extract note frequencies + individual decay rates
   - Use damping coefficients to distinguish overlapping notes
   - Model per-note dynamics

2. **Variable-Q Wavelet** → Multi-pitch detection with temporal context
   - Phase coherence for harmonic grouping
   - Amplitude modulation for note segmentation

3. **Gammatone Filterbank** → Onset detection + timbre features
   - Envelope peaks for note onsets
   - Filter-specific dynamics for instrument classification

---

## Mathematical Background

### Laplace Transform Intuition
```
L{f(t)} = ∫₀^∞ f(t)e^(-st) dt,  s = σ + jω
```
- **σ (real part)**: Growth/decay rate
- **ω (imaginary part)**: Oscillation frequency

### Why Windowing Is Problematic
- Laplace assumes causality from t=0 to ∞
- Windowing truncates signals arbitrarily
- Short-time methods are pragmatic approximations

### Connection to Z-Transform
For digital implementation:
```
z = e^(sT)  where T = 1/fs
```
Prony method finds poles in z-domain, converts back to s-domain.

---

## Limitations

### Prony Analysis
- **Sensitive to noise**: Requires careful preprocessing
- **Model order selection**: Too high = spurious poles, too low = missing components
- **Edge effects**: Windowing introduces artifacts

### Variable-Q Wavelet
- **Computational cost**: Higher than STFT
- **No explicit damping**: Must infer from phase/amplitude derivatives
- **Limited temporal resolution**: At low frequencies

### Gammatone Filterbank
- **Not invertible**: Can't reconstruct original signal perfectly
- **Fixed filter shapes**: Less flexible than adaptive methods
- **Many filters needed**: For good frequency coverage

---

## Future Extensions

1. **Sparse Prony**: Use L1 regularization for robust component selection
2. **Adaptive VQT**: Dynamic Q-factor based on signal characteristics
3. **Cascaded Gammatone**: Multiple filterbank stages for finer resolution
4. **Hybrid Methods**: Combine Prony peaks with VQT context
5. **Neural Integration**: Use features as input to deep learning models

---

## References

### Prony Method
- Parks, T.W., & Burrus, C.S. (1987). *Digital Filter Design*
- Kumaresan, R., & Tufts, D.W. (1982). "Estimating the parameters of exponentially damped sinusoids"

### Variable-Q Transform
- Schörkhuber, C., & Klapuri, A. (2010). "Constant-Q transform toolbox for music processing"
- Librosa documentation: https://librosa.org/doc/main/generated/librosa.vqt.html

### Gammatone Filterbank
- Patterson, R.D., et al. (1988). "Complex sounds and auditory images"
- Slaney, M. (1993). "An Efficient Implementation of the Patterson-Holdsworth Auditory Filter Bank"

---

## Contact & Contributions

For music technology applications, this implementation provides a solid foundation for:
- Transcription systems with envelope awareness
- Source separation with decay-based clustering
- Audio synthesis with physically-inspired models

Feel free to extend and modify for your specific use case!

---

## License

MIT License - Use freely with attribution
