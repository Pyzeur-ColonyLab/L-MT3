# Quick Start Guide - Laplace Audio Analysis

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Basic Demo
```bash
python laplace_audio_analysis.py
```

This generates visualizations for all three methods using a test signal.

### Step 3: Run Advanced Transcription Demo
```bash
python advanced_examples.py
```

This demonstrates a complete polyphonic transcription pipeline combining all methods.

---

## 📊 What You Get

### From `laplace_audio_analysis.py`:
- **prony_analysis.png** - Frequency + damping extraction
- **vqt_analysis.png** - Multi-resolution time-frequency analysis
- **gammatone_analysis.png** - Perceptual filterbank with envelope tracking

### From `advanced_examples.py`:
- **advanced_transcription_demo.png** - Complete audio-to-MIDI pipeline with:
  - Onset detection (Gammatone)
  - Pitch tracking (VQT)
  - Decay analysis (Prony)
  - Piano-roll visualization

---

## 🎵 Using Your Own Audio

### Simple Analysis
```python
import librosa
from laplace_audio_analysis import PronyAnalyzer, VariableQWavelet, GammatoneFilterbank

# Load your audio file
audio, sr = librosa.load('your_music.wav', sr=22050, duration=5)

# Pick a method and analyze
prony = PronyAnalyzer(n_components=10, model_order=30)
results = prony.analyze(audio, sr)
```

### Complete Transcription Pipeline
```python
from advanced_examples import combine_methods_for_transcription

audio, sr = librosa.load('your_music.wav', sr=22050)
results = combine_methods_for_transcription(audio, sr)

# Access detected notes
for note in results['notes']:
    print(f"MIDI {note['midi']}: {note['onset']:.2f}s - {note['offset']:.2f}s")
    if note['damping']:
        print(f"  Decay rate: {note['damping']:.2f}")
```

---

## 🎯 Which Method to Use?

| Task | Best Method | Why |
|------|-------------|-----|
| **Piano transcription** | Prony + VQT | Capture decay differences + pitch tracking |
| **Onset detection** | Gammatone | Robust to noise, perceptually accurate |
| **Pitch tracking** | VQT | Multi-resolution, handles vibrato |
| **Timbre analysis** | Gammatone | Frequency-dependent envelopes |
| **Source separation** | Prony | Explicit decay rates distinguish sources |

---

## ⚙️ Parameter Tuning Tips

### Prony Analysis
- **High noise?** → Lower `model_order` (15-25)
- **Missing notes?** → Higher `n_components` (15-20)
- **Spurious peaks?** → Shorter `win_length` (1024)

### Variable-Q Wavelet
- **Piano/guitar?** → `bins_per_octave=36` (finer resolution)
- **Rhythm analysis?** → Lower `fmin` (20 Hz)
- **Harmonic content?** → Increase `n_bins` (96+)

### Gammatone Filterbank
- **More detail?** → `n_filters=128`
- **Bass-heavy?** → Lower `fmin` (20 Hz)
- **Bright sounds?** → Higher `fmax` (12000 Hz)

---

## 🔬 For Audio-to-MIDI Development

The combined method in `advanced_examples.py` gives you:

1. **Onset times** with frequency information
2. **Pitch trajectories** over time
3. **Decay characteristics** per note

This is richer than standard STFT because you get:
- ✅ **When** each note starts (onset detection)
- ✅ **What** pitch over time (trajectory tracking)
- ✅ **How** it decays (Laplace-style damping)

Perfect for distinguishing:
- Overlapping notes with different decay rates
- Same pitch played on different instruments
- Notes with different articulations (staccato vs legato)

---

## 📖 Next Steps

1. **Read README.md** for comprehensive documentation
2. **Check the code** - all methods are well-commented
3. **Experiment** with your own audio files
4. **Combine methods** for your specific use case

---

## 💡 Key Insight

Traditional STFT assumes infinite sinusoids. Real music has:
- Attack/decay envelopes
- Different damping rates
- Transient behavior

These three methods capture this temporal richness, making them ideal for:
- Polyphonic transcription
- Source separation
- Instrument classification
- Music information retrieval

---

## 🤝 Questions or Issues?

All code is MIT licensed. Feel free to modify and extend for your projects!

For music tech applications, this provides a solid foundation beyond standard Fourier analysis.
