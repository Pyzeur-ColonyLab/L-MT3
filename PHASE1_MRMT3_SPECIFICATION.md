# Phase 1: Inference-Only Enhancement of MR-MT3 with Laplace Features

## Technical Specification Document

**Version:** 1.0  
**Date:** November 2024  
**Target Model:** [MR-MT3](https://github.com/gudgud96/MR-MT3) by gudgud96  
**Baseline Paper:** [arXiv:2403.10024](https://arxiv.org/abs/2403.10024)

---

## Executive Summary

This document specifies a **post-processing enhancement pipeline** for MR-MT3 that leverages Laplace transform-inspired audio features to further reduce instrument leakage beyond what MR-MT3's memory retention mechanism achieves. The enhancement works **at inference time without retraining**, making it fast to implement and test.

**Key Innovation:**  
While MR-MT3 uses memory retention across temporal segments, we add **decay-based memory selection** and **timbre consistency post-processing** to consolidate fragmented instruments that share similar physical characteristics.

**Expected Improvements:**
- 10-25% additional reduction in instrument leakage ratio (φ)
- 15-30% improvement in decay consistency (new metric)
- 5-10% boost in instrument detection F1
- Zero retraining required (inference-only)

---

## 1. Background & Motivation

### 1.1 MR-MT3 Architecture Overview

MR-MT3 introduces a memory retention mechanism that aggregates tokens transcribed from the previous segment and concatenates them to the encoder outputs for cross-attention during autoregressive token sampling of the current segment.

**Key Components:**
1. **Encoder:** T5-based encoder processes audio spectrograms
2. **Memory Module:** Retains previous segment tokens for context
3. **Decoder:** Autoregressive token generation with memory-augmented cross-attention
4. **Prior Token Sampling:** Samples relevant prior tokens from past segments
5. **Token Shuffling:** Data augmentation during training

### 1.2 Remaining Issues in MR-MT3

Despite MR-MT3's improvements, instrument leakage persists where transcriptions are fragmented across different instruments. Analysis reveals:

**Issue 1: Memory Selection is Time-Based Only**
- MR-MT3 uses temporal proximity for memory retention
- Doesn't consider whether memories are from the *same instrument*
- Result: Piano from 5s ago might inform violin transcription now

**Issue 2: No Physical Timbre Validation**
- Token-based approach lacks acoustic grounding
- Multiple "piano" tracks might have different decay characteristics
- No post-hoc validation of instrument assignments

**Issue 3: Limited Spectral Context**
- Spectrogram input is Fourier-based (assumes eternal sinusoids)
- Doesn't capture exponential decay rates (key instrument signature)
- Missing perceptual filterbank information

### 1.3 Why Laplace Features?

Traditional **Fourier Transform** assumes signals are eternal sinusoids:
```
X(ω) = ∫ x(t)e^(-jωt) dt
```

**Laplace Transform** naturally captures transient behavior:
```
X(s) = ∫ x(t)e^(-st) dt,  where s = σ + jω
```

The damping factor σ reveals:
- **Attack time:** How fast notes begin
- **Decay rate:** How fast notes fade
- **Sustain characteristics:** Steady-state behavior

These are **instrument fingerprints** that Fourier analysis misses.

---

## 2. Proposed Enhancement Architecture

### 2.1 Pipeline Overview

```
Audio Input (16kHz)
    ↓
┌─────────────────────────────────┐
│  MR-MT3 Vanilla Transcription   │
│  - Memory retention              │
│  - Prior token sampling          │
│  - Autoregressive decoding       │
└─────────────────────────────────┘
    ↓
Raw MIDI (with leakage)
    ↓
┌─────────────────────────────────┐
│  Laplace Feature Extraction      │
│  (operates on raw audio)         │
│                                  │
│  1. Prony Analysis → Decay rates │
│  2. Variable-Q Transform → Spectral envelope │
│  3. Gammatone Filterbank → Perceptual features │
└─────────────────────────────────┘
    ↓
Instrument-level features
    ↓
┌─────────────────────────────────┐
│  Decay-Based Consolidation       │
│  - Group instruments by decay    │
│  - Merge notes with similar σ    │
│  - Reduce leakage                │
└─────────────────────────────────┘
    ↓
Consolidated MIDI
    ↓
┌─────────────────────────────────┐
│  Timbre-Based Refinement         │
│  - Validate instrument classes   │
│  - Fix misclassifications        │
│  - Use spectral centroids        │
└─────────────────────────────────┘
    ↓
Enhanced MIDI (less leakage)
```

### 2.2 Integration Points with MR-MT3

**Critical:** This pipeline runs **after** MR-MT3 inference, not during.

```python
# Pseudo-code for integration
from mr_mt3.inference import infer
from laplace_enhancement import enhance_transcription

# Step 1: Run vanilla MR-MT3
audio = load_audio("song.wav", sr=16000)
midi_raw = infer(audio, model=mr_mt3_checkpoint)

# Step 2: Apply Laplace enhancement
midi_enhanced = enhance_transcription(
    midi=midi_raw,
    audio=audio,
    sr=16000
)

# Result: Reduced instrument leakage
```

**Key Point:** We don't modify MR-MT3's weights or architecture. This is pure post-processing.

---

## 3. Laplace Feature Extraction

### 3.1 Prony Analysis for Decay Rates

**Purpose:** Extract exponential decay components from audio segments.

**Method:** Prony's method models signals as sums of complex exponentials:
```
x[n] = Σ(k=1 to K) a_k * exp(s_k * n)
```

Where `s_k = σ_k + jω_k`:
- `σ_k` → Decay rate (negative for decaying signals)
- `ω_k` → Frequency (radians/sample)

**Implementation:**
```python
from laplace_audio_analysis import PronyAnalyzer

prony = PronyAnalyzer(n_components=10, model_order=30)

# For each note in each instrument track
for instrument in midi.instruments:
    decay_rates = []
    
    for note in instrument.notes:
        segment = audio[note.start*sr : note.end*sr]
        result = prony.analyze(segment, sr, hop_length=256, win_length=1024)
        
        # Extract damping factors (σ)
        damping = [d for frame in result['damping'] for d in frame if d < 0]
        if damping:
            decay_rates.append(np.median(damping))
    
    instrument_decay = np.mean(decay_rates)
```

**Output:**
- Each instrument gets a characteristic decay rate
- Similar decay → likely same instrument
- Different decay → different instruments

### 3.2 Variable-Q Transform for Spectral Envelope

**Purpose:** Multi-resolution time-frequency representation with logarithmic frequency spacing (matches musical pitch).

**Advantages over STFT:**
- Better frequency resolution at low frequencies (bass, cello)
- Better time resolution at high frequencies (cymbals, hi-hats)
- Matches human perception and musical scales

**Implementation:**
```python
from laplace_audio_analysis import VariableQWavelet

vqt = VariableQWavelet(fmin=55, n_bins=72, bins_per_octave=12)

for instrument in midi.instruments:
    spectral_centroids = []
    
    for note in instrument.notes:
        segment = audio[note.start*sr : note.end*sr]
        vqt_mag, times, freqs = vqt.analyze(segment, sr)
        
        # Compute spectral centroid
        profile = np.mean(np.abs(vqt_mag), axis=1)
        centroid = np.sum(freqs * profile) / np.sum(profile)
        spectral_centroids.append(centroid)
    
    instrument_centroid = np.mean(spectral_centroids)
```

**Output:**
- Spectral centroid: brightness of timbre (Hz)
- Low centroid (<500 Hz): dark instruments (bass, low strings)
- High centroid (>1500 Hz): bright instruments (trumpet, violin)

### 3.3 Gammatone Filterbank for Perceptual Features

**Purpose:** Mimic human auditory system's frequency decomposition.

**Why it matters:**
- Instruments sound different because our ears perceive them differently
- Gammatone filters model cochlear response
- Captures envelope modulation (vibrato, tremolo)

**Implementation:**
```python
from laplace_audio_analysis import GammatoneFilterbank

gammatone = GammatoneFilterbank(n_filters=48, fmin=50, fmax=4000)

for instrument in midi.instruments:
    envelope_features = []
    
    for note in instrument.notes:
        segment = audio[note.start*sr : note.end*sr]
        filtered, envelopes = gammatone.analyze(segment, sr)
        
        # Extract temporal envelope characteristics
        attack_time = gammatone.estimate_attack_time(envelopes)
        envelope_features.append(attack_time)
    
    instrument_attack = np.mean(envelope_features)
```

**Output:**
- Attack time: how fast notes begin (ms)
- Fast attack: percussive (piano, guitar)
- Slow attack: sustained (organ, strings)

---

## 4. Decay-Based Consolidation Algorithm

### 4.1 Problem Statement

MR-MT3 may output:
```
Track 1: Piano (notes A, B, C) - decay_rate = -2000
Track 2: Piano (notes D, E, F) - decay_rate = -2100  ← Leakage!
Track 3: Piano (notes G, H, I) - decay_rate = -1950  ← Leakage!
```

**Goal:** Merge these into one coherent piano track.

### 4.2 Similarity Metrics

For each pair of instruments i, j, compute:

**1. Decay Similarity**
```
decay_diff = |mean_decay[i] - mean_decay[j]|
decay_avg = (|mean_decay[i]| + |mean_decay[j]|) / 2
decay_sim = 1 - (decay_diff / decay_avg)
```

**2. Spectral Similarity**
```
spectral_diff = |centroid[i] - centroid[j]|  (Hz)
```

**3. MIDI Class Consistency**
```
midi_class[i] = program[i] // 8
midi_class[j] = program[j] // 8
same_family = (midi_class[i] == midi_class[j])
```

### 4.3 Merging Rules

Merge instruments i and j if **ALL** conditions met:

1. `same_family == True` (e.g., both keyboards, both strings)
2. `decay_sim > 0.80` (within 20% decay difference)
3. `spectral_diff < 200 Hz` (similar brightness)
4. Neither is percussion (`is_drum == False`)

**Threshold Tuning:**
- Conservative (fewer merges): `decay_sim > 0.90`, `spectral_diff < 150`
- Aggressive (more merges): `decay_sim > 0.70`, `spectral_diff < 300`

### 4.4 Implementation

```python
def consolidate_by_decay(midi, audio, sr):
    """
    Merge instruments with similar decay characteristics
    """
    # Extract features for all instruments
    features = {}
    for i, inst in enumerate(midi.instruments):
        features[i] = extract_decay_features(inst, audio, sr)
    
    # Build similarity graph
    groups = {}
    processed = set()
    
    for i in features.keys():
        if i in processed:
            continue
        
        group = [i]
        processed.add(i)
        
        for j in features.keys():
            if j in processed or j == i:
                continue
            
            if should_merge(features[i], features[j]):
                group.append(j)
                processed.add(j)
        
        groups[i] = group
    
    # Create merged MIDI
    consolidated = pretty_midi.PrettyMIDI()
    for primary_idx, group_indices in groups.items():
        merged_instrument = merge_notes(
            [midi.instruments[idx] for idx in group_indices]
        )
        consolidated.instruments.append(merged_instrument)
    
    return consolidated
```

---

## 5. Timbre-Based Refinement

### 5.1 Purpose

Even after consolidation, some instrument assignments may be wrong:
- "Violin" track with piano-like decay → should be piano
- "Piano" track with bright timbre → might be guitar

### 5.2 Heuristic Rules

Based on extracted features, apply corrections:

**Rule 1: Dark, Harmonic → Bass**
```
if centroid < 300 and harmonic_ratio > 2.0:
    program = 32  # Acoustic Bass
```

**Rule 2: Mid-range, Strong Harmonics → Strings**
```
if 300 < centroid < 800 and harmonic_ratio > 3.0:
    program = 40  # Violin
```

**Rule 3: Bright, Rich Spectrum → Brass**
```
if centroid > 1500 and harmonic_ratio > 2.5:
    program = 56  # Trumpet
```

**Rule 4: Fast Attack, Percussive → Guitar/Piano**
```
if attack_time < 10ms:
    if spectral_centroid > 1000:
        program = 24  # Acoustic Guitar
    else:
        program = 0   # Acoustic Grand Piano
```

### 5.3 Future Enhancement (Phase 2)

Replace heuristics with **trained classifier**:
```python
from instrument_recognition import InstrumentClassifier

classifier = InstrumentClassifier()
classifier.fit(training_data)

# For each instrument
predicted_class = classifier.predict(extract_features(audio, notes))
instrument.program = class_to_midi_program(predicted_class)
```

---

## 6. Evaluation Metrics

### 6.1 Standard Metrics (from MR-MT3 paper)

**1. Multi-Instrument Onset F1**
- Measures note-level accuracy
- Expected: No change (we don't add/remove notes)
- Baseline: 67.3% (MIDI Class, Slakh2100)

**2. Instrument Leakage Ratio (φ)**
```
φ = n_predicted_instruments / n_true_instruments
```
- φ = 1.0 → Perfect
- φ > 1.0 → Over-prediction (leakage)
- φ < 1.0 → Under-prediction
- **Baseline (MR-MT3):** 1.24
- **Target (Phase 1):** 1.00-1.10

**3. Instrument Detection F1**
- Precision/Recall of detected instrument classes
- **Baseline:** 39.1%
- **Target:** 42-45%

### 6.2 New Metrics (Laplace-Specific)

**4. Decay Consistency Score**

For each instrument track, measure variance of decay rates:
```
decay_rates = [σ_1, σ_2, ..., σ_n]
cv = std(decay_rates) / |mean(decay_rates)|
consistency = 1 / (1 + cv)
```

- 1.0 = Perfect consistency (all notes have same decay)
- 0.5 = Moderate consistency
- 0.0 = No consistency (random decay rates)

**Why it matters:** Real instruments have consistent decay. High consistency → cleaner transcription.

**5. Timbre Homogeneity**

Within each instrument track, measure spectral variance:
```
centroids = [c_1, c_2, ..., c_n]
homogeneity = 1 - (std(centroids) / mean(centroids))
```

**Expected:**
- Vanilla MR-MT3: 0.45-0.65
- Enhanced: 0.70-0.85

---

## 7. Implementation Plan

### 7.1 Phase 1A: Standalone Pipeline (Week 1)

**Deliverables:**
1. `phase1_mrmt3_enhancement.py` - Main script
2. Integration with existing `laplace_audio_analysis.py`
3. `test_phase1_mrmt3.py` - Automated test suite
4. Documentation and usage guide

**Timeline:**
- Day 1-2: Adapt existing code for MR-MT3 MIDI format
- Day 3-4: Implement consolidation algorithm
- Day 5-6: Implement refinement and metrics
- Day 7: Testing and documentation

### 7.2 Phase 1B: Evaluation (Week 2)

**Tasks:**
1. Download MR-MT3 pretrained checkpoint from [HuggingFace](https://huggingface.co/gudgud1014/MR-MT3/tree/main)
2. Run on Slakh2100 test set (145 tracks)
3. Compute all metrics (φ, F1, decay consistency)
4. Compare: Vanilla MT3 → MR-MT3 → MR-MT3 + Laplace
5. Statistical analysis and visualization

**Dataset:** Slakh2100 (follow MR-MT3 preprocessing)
```bash
# From MR-MT3 repo
python3 resample.py          # 44.1kHz → 16kHz
python3 midi_script.py       # Fix octave errors
python3 tools/generate_inst_names.py
```

### 7.3 Success Criteria

**Minimum Viable:**
- ✅ φ improves by >5% (1.24 → 1.18)
- ✅ Decay consistency >0.65
- ✅ No degradation in onset F1

**Target:**
- ✅ φ improves by >10% (1.24 → 1.12)
- ✅ Decay consistency >0.75
- ✅ Instrument detection F1 +3pp

**Excellent:**
- ✅ φ improves by >15% (1.24 → 1.05)
- ✅ Decay consistency >0.80
- ✅ Instrument detection F1 +5pp

---

## 8. Integration with MR-MT3 Repository

### 8.1 Minimal Code Changes

**Option 1: Wrapper Script (Recommended)**

No changes to MR-MT3 codebase. Create wrapper:
```python
# enhance_mrmt3.py
import sys
sys.path.append('/path/to/MR-MT3')

from inference import transcribe_audio
from laplace_enhancement import enhance_transcription

def enhanced_mrmt3_inference(audio_path):
    # Use MR-MT3 as-is
    midi_raw = transcribe_audio(audio_path)
    
    # Apply Laplace enhancement
    audio, sr = librosa.load(audio_path, sr=16000)
    midi_enhanced = enhance_transcription(midi_raw, audio, sr)
    
    return midi_enhanced
```

**Option 2: Plugin Architecture**

Modify `test.py` to add optional post-processing:
```python
# In MR-MT3/test.py
if args.use_laplace_enhancement:
    from laplace_enhancement import enhance_transcription
    midi = enhance_transcription(midi, audio, sr)
```

### 8.2 Configuration

Add to MR-MT3 config:
```yaml
# config/enhancement.yaml
laplace:
  enabled: true
  consolidation:
    decay_threshold: 0.80
    spectral_threshold: 200  # Hz
  refinement:
    use_heuristics: true
    use_classifier: false  # Phase 2
```

---

## 9. Comparison with MR-MT3 Baseline

### 9.1 MR-MT3 Improvements Over MT3

From the paper:

| Method | Onset F1 (MIDI Class) | φ (Leakage) | Inst. Det. F1 |
|--------|----------------------|-------------|---------------|
| MT3 (baseline) | 62.5% | ~1.65 | 35.2% |
| MR-MT3 (memory) | 67.3% | 1.24 | 39.1% |
| **Improvement** | **+4.8pp** | **-25%** | **+3.9pp** |

**Key MR-MT3 Innovations:**
1. Memory retention across segments
2. Prior token sampling
3. Token shuffling (training)

### 9.2 Phase 1 Expected Cumulative Results

| Method | Onset F1 | φ | Inst. Det. F1 | Decay Consist. |
|--------|----------|---|---------------|----------------|
| MT3 | 62.5% | 1.65 | 35.2% | N/A |
| MR-MT3 | 67.3% | 1.24 | 39.1% | N/A |
| **MR-MT3 + Laplace** | **67.3%** | **1.05-1.12** | **42-45%** | **0.75-0.80** |
| **Δ from MR-MT3** | **0pp** | **-10-15%** | **+3-6pp** | **NEW** |

**Why Onset F1 unchanged?**
- We don't modify note detection, only instrument grouping
- Onset accuracy is already at MR-MT3's level
- Focus is on reducing leakage, not finding new notes

### 9.3 Computational Cost

**MR-MT3 Inference:**
- ~5-10 seconds per 30s audio (GPU)
- Memory: ~2GB VRAM

**Phase 1 Enhancement:**
- +2-3 seconds per 30s audio (CPU)
- Memory: ~500MB RAM
- **Total overhead:** ~30-40% increase in inference time

**Why acceptable:**
- No retraining (saves days/weeks)
- Significant quality improvement
- Can be parallelized per-instrument

---

## 10. Risk Analysis & Mitigation

### 10.1 Potential Issues

**Risk 1: Prony Fails on Noisy Audio**
- **Symptom:** Empty results, no decay estimates
- **Impact:** Consolidation falls back to VQT+Gammatone only
- **Mitigation:** Graceful degradation, still functional
- **Probability:** Medium (10-20% of real-world audio)

**Risk 2: Over-Consolidation**
- **Symptom:** Merges distinct instruments (e.g., violin + cello)
- **Impact:** Under-prediction, φ < 1.0
- **Mitigation:** Tune thresholds conservatively, add manual override
- **Probability:** Low (5%) with proper tuning

**Risk 3: No Improvement on Clean Transcriptions**
- **Symptom:** MR-MT3 already very accurate on some tracks
- **Impact:** Phase 1 has no effect (but doesn't hurt)
- **Mitigation:** Focus on tracks with known leakage
- **Probability:** Medium (20-30% of tracks may be near-optimal)

**Risk 4: Computational Bottleneck**
- **Symptom:** Prony analysis slow on long files
- **Impact:** Inference time increases significantly
- **Mitigation:** Batch processing, parallel computation, sampling
- **Probability:** Low with optimization

### 10.2 Mitigation Strategies

1. **Adaptive Thresholding:** Adjust consolidation aggressiveness per track
2. **Feature Caching:** Save Laplace features to disk for reuse
3. **Partial Enhancement:** Apply only to tracks with high leakage (φ > 1.3)
4. **Hybrid Mode:** Use heuristics first, fall back to MR-MT3 if uncertain

---

## 11. Future Work (Phase 2+)

### 11.1 Phase 2: Fine-Tuning with Laplace Encoder

**Approach:**
1. Freeze MR-MT3 weights
2. Add trainable Laplace feature encoder
3. Concatenate Laplace embeddings to MR-MT3's encoder output
4. Fine-tune on Slakh2100

**Expected:**
- +5-8% additional improvement
- Better integration of Laplace features
- Learned weighting of decay vs spectral vs envelope

**Timeline:** 2-3 weeks training + evaluation

### 11.2 Phase 3: End-to-End Architecture

**Approach:**
1. Replace spectrogram with Laplace-domain representation
2. Train from scratch with modified loss:
   - L_onset (note-level accuracy)
   - L_decay (decay consistency)
   - L_leakage (penalize instrument over-prediction)

**Expected:**
- 10-15% improvement over MR-MT3
- New SOTA for multi-instrument transcription
- Publication-quality results

**Timeline:** 1-2 months (full research project)

---

## 12. Deliverables Checklist

### 12.1 Code
- [ ] `phase1_mrmt3_enhancement.py` - Main enhancement script
- [ ] `test_phase1_mrmt3.py` - Automated test suite
- [ ] `evaluate_slakh2100.py` - Batch evaluation script
- [ ] `visualize_results.py` - Comparison plots
- [ ] Integration guide with MR-MT3 repo

### 12.2 Documentation
- [ ] Technical specification (this document)
- [ ] User guide with examples
- [ ] API documentation
- [ ] Troubleshooting guide
- [ ] Parameter tuning guide

### 12.3 Evaluation
- [ ] Results on Slakh2100 test set
- [ ] Comparison tables (MT3 vs MR-MT3 vs Enhanced)
- [ ] Ablation study (which features matter most?)
- [ ] Audio examples and visualizations
- [ ] Statistical significance tests

### 12.4 Reproducibility
- [ ] Requirements.txt with exact versions
- [ ] Docker container (optional)
- [ ] Pre-computed features on Slakh2100 (optional)
- [ ] Pretrained models/checkpoints
- [ ] Random seeds and config files

---

## 13. Getting Started

### 13.1 Quick Setup

```bash
# Clone MR-MT3
git clone https://github.com/gudgud96/MR-MT3.git
cd MR-MT3

# Install MR-MT3 dependencies
conda create -n mrmt3 python=3.10
conda activate mrmt3
pip install -r requirements.txt

# Add Laplace enhancement
cd ..
git clone [your-laplace-repo]
cd laplace-enhancement
pip install -e .

# Download pretrained MR-MT3
wget https://huggingface.co/gudgud1014/MR-MT3/resolve/main/mr_mt3_checkpoint.pt

# Test on single file
python phase1_mrmt3_enhancement.py \
    --audio example.wav \
    --model ../MR-MT3/checkpoints/mr_mt3_checkpoint.pt \
    --output output.mid \
    --compare
```

### 13.2 Running on Slakh2100

```bash
# Download and prepare Slakh2100
bash scripts/download_slakh2100.sh
python3 resample.py
python3 midi_script.py

# Run evaluation
python evaluate_slakh2100.py \
    --data_dir ./slakh2100_redux/test \
    --model_checkpoint ../MR-MT3/checkpoints/mr_mt3_checkpoint.pt \
    --output_dir ./results/phase1 \
    --use_laplace_enhancement
```

### 13.3 Expected Timeline

**Week 1:** Setup + Implementation
- Day 1-2: Environment setup, download models/data
- Day 3-4: Implement consolidation algorithm
- Day 5-7: Testing and bug fixes

**Week 2:** Evaluation
- Day 1-3: Run on full Slakh2100 test set
- Day 4-5: Compute metrics and generate plots
- Day 6-7: Analysis and documentation

**Week 3:** Refinement (if needed)
- Parameter tuning based on results
- Edge case handling
- Performance optimization

---

## 14. Contact & Support

**Primary Developer:** [Your Name]  
**Email:** [Your Email]  
**Repository:** [Link to your repo]

**References:**
- MR-MT3 Paper: https://arxiv.org/abs/2403.10024
- MR-MT3 Code: https://github.com/gudgud96/MR-MT3
- Slakh2100 Dataset: http://www.slakh.com

**Citation:**
```bibtex
@article{tan2024mr,
  title={MR-MT3: Memory Retaining Multi-Track Music Transcription to Mitigate Instrument Leakage},
  author={Tan, Hao Hao and Cheuk, Kin Wai and Cho, Taemin and Liao, Wei-Hsiang and Mitsufuji, Yuki},
  journal={arXiv preprint arXiv:2403.10024},
  year={2024}
}
```

---

## Appendix A: Parameter Reference

```python
# Consolidation thresholds
DECAY_SIMILARITY_THRESHOLD = 0.80      # [0.7-0.9]
SPECTRAL_DIFF_THRESHOLD = 200          # Hz [150-300]
MIDI_CLASS_MATCH_REQUIRED = True       # [True/False]

# Prony analysis
PRONY_N_COMPONENTS = 10                # [5-20]
PRONY_MODEL_ORDER = 30                 # [20-50]
PRONY_MIN_NOTE_DURATION = 0.05         # seconds

# VQT parameters
VQT_FMIN = 55                          # Hz (A1)
VQT_N_BINS = 72                        # 6 octaves
VQT_BINS_PER_OCTAVE = 12               # semitones

# Gammatone filterbank
GAMMATONE_N_FILTERS = 48               # [32-64]
GAMMATONE_FMIN = 50                    # Hz
GAMMATONE_FMAX = 4000                  # Hz

# Refinement heuristics
DARK_TIMBRE_THRESHOLD = 300            # Hz
MID_TIMBRE_THRESHOLD = 800             # Hz
BRIGHT_TIMBRE_THRESHOLD = 1500         # Hz
HARMONIC_RATIO_THRESHOLD = 2.0         # [1.5-3.0]
FAST_ATTACK_THRESHOLD = 0.010          # seconds
```

---

**Document Status:** Draft v1.0  
**Next Review:** After Week 1 implementation  
**Approval:** Pending initial results
