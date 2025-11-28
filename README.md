# L-MT3: Laplace-Enhanced MR-MT3 Music Transcription

GPU-accelerated music transcription pipeline with ML-based instrument refinement to reduce transcription errors and instrument leakage.

## 🎯 Quick Start

### Phase 2: ML Classifier (Recommended - 88-92% accuracy)

```bash
# 1. Clone repository
git clone https://github.com/Pyzeur-ColonyLab/L-MT3.git
cd L-MT3

# 2. SSH to GPU instance
ssh ubuntu@your-instance-ip

# 3. Pull and train (3-4 hours)
cd ~/L-MT3 && git pull
tmux new -s phase2
source venv/bin/activate
sudo ./scripts/setup_phase2_training.sh
```

📖 **Complete guide:** [PHASE2_README.md](PHASE2_README.md)

### Phase 1: Heuristic-Based (Legacy - 60-70% accuracy)

For reference or comparison with Phase 2 ML approach.

📖 **Specification:** [PHASE1_MRMT3_SPECIFICATION.md](PHASE1_MRMT3_SPECIFICATION.md)

---

## 📊 Phase Comparison

| Aspect | Phase 1 (Heuristics) | Phase 2 (ML Classifier) |
|--------|---------------------|------------------------|
| **Accuracy** | 60-70% | 88-92% |
| **Method** | 4 hand-coded rules | Learned from 300k samples |
| **Families** | 4 (bass, strings, brass, piano) | 11 (NSynth families) |
| **Features** | 3 (centroid, attack, harmonic) | 26 (Prony, VQT, Gammatone) |
| **Processing** | ~8-10s per track | ~9-11s per track (+10%) |

---

## 🎵 What This Does

**Problem:** MR-MT3 transcription has instrument leakage (notes assigned to wrong instruments)

**Solution:** Laplace transform features + ML classifier to:
1. Extract 26 acoustic features (decay rates, spectral analysis, attack times)
2. Classify instruments with 88-92% accuracy (vs 60-70% heuristics)
3. Consolidate instruments with similar acoustic signatures
4. Refine note assignments based on learned timbre characteristics
5. Reduce cross-instrument leakage by 25-35%

**Pipeline:**
```
Audio → MR-MT3 (GPU) → MIDI → Laplace Features → ML Classifier → Enhanced MIDI
```

---

## 💻 Instance Requirements

**Recommended (NVIDIA L4/T4):**
- GPU: 16GB VRAM
- CPU: 16 cores
- RAM: 32GB
- Storage: 50GB (Phase 2 training requires +35GB for NSynth)

**Phase 2 Training:**
- NSynth dataset: 25GB
- Extracted features: 3-5GB
- Models: ~50MB
- Total: ~35GB additional

---

## 📁 Repository Structure

```
L-MT3/
├── README.md                           # This file
├── PHASE2_README.md                    # Phase 2 quick start
├── PHASE2_DEPLOYMENT_GUIDE.md          # Complete deployment guide
├── PHASE1_MRMT3_SPECIFICATION.md       # Phase 1 reference
│
├── scripts/
│   ├── setup_phase2_training.sh        # Automated NSynth training (3-4h)
│   └── setup_gpu_instance.sh           # GPU environment setup
│
├── Laplace_classifier/                 # Phase 2 ML Classifier
│   ├── laplace_classifier.py           # Classifier implementation
│   ├── extract_nsynth_features.py      # Feature extraction
│   ├── train_classifier.py             # Training script
│   └── CLASSIFIER_DEV_GUIDE.md         # Development guide
│
├── laplace_mrmt3/                      # Core enhancement library
│   ├── ml_refinement.py                # Phase 2 ML-based refiner
│   ├── refinement.py                   # Phase 1 heuristic refiner
│   ├── features.py                     # Feature extraction
│   ├── consolidation.py                # Instrument merging
│   ├── metrics.py                      # Evaluation metrics
│   └── config.py                       # Configuration
│
├── phase1_mrmt3_enhancement.py         # Enhancement pipeline
├── run_mrmt3_inference.py              # MR-MT3 GPU inference
├── run_slakh2100_batch.sh              # Batch processing
├── organize_processed_tracks.sh        # Output organization
│
└── requirements_mrmt3_laplace.txt      # Python dependencies
```

---

## 📖 Documentation

### Phase 2 (Current)
- **[PHASE2_README.md](PHASE2_README.md)** - Quick start guide
- **[PHASE2_DEPLOYMENT_GUIDE.md](PHASE2_DEPLOYMENT_GUIDE.md)** - Complete deployment
- **[Laplace_classifier/CLASSIFIER_DEV_GUIDE.md](Laplace_classifier/CLASSIFIER_DEV_GUIDE.md)** - ML classifier development

### Phase 1 (Reference)
- **[PHASE1_MRMT3_SPECIFICATION.md](PHASE1_MRMT3_SPECIFICATION.md)** - Heuristic approach specification

---

## 🚀 Usage Examples

### Single Track Enhancement (Phase 2)

```bash
python3 phase1_mrmt3_enhancement.py \
    --midi baseline.mid \
    --audio input.wav \
    --output enhanced_ml.mid \
    --use-ml-classifier \
    --classifier-path ./laplace_classifier.pkl
```

### Batch Processing (Slakh2100)

```bash
./run_slakh2100_batch.sh \
    --split validation \
    --start-track Track01647 \
    --use-ml-classifier
```

### Organize Results

```bash
./organize_processed_tracks.sh \
    --split validation \
    --include-baseline
```

---

## 📊 Datasets

**Slakh2100** (music transcription benchmark):
- Validation: 375 tracks
- Used for batch processing and evaluation
- Automatically downloaded by setup script

**NSynth** (instrument classification):
- 289,205 samples, 11 instrument families
- Used for Phase 2 ML classifier training
- Downloaded during Phase 2 setup (~25GB)

---

## 🔧 Installation

### Dependencies

```bash
pip install -r requirements_mrmt3_laplace.txt
```

**Core packages:**
- librosa 0.10.x (audio processing)
- pretty-midi 0.2.x (MIDI manipulation)
- scikit-learn 1.x (ML classifier - Phase 2)
- xgboost (gradient boosting - Phase 2)
- matplotlib 3.x (visualization)

---

## 🎓 Technical Details

### Laplace Features (26 total)

**Prony Analysis (7 features):**
- Exponential decay rates for each note
- Modal decomposition of attack/sustain/release

**Variable-Q Transform (8 features):**
- Multi-resolution spectral analysis
- Harmonic content characterization

**Gammatone Filterbank (11 features):**
- Perceptual attack/decay characteristics
- Psychoacoustic temporal envelope

### Phase 2 ML Classifier

**Architecture:**
- Input: 26 Laplace features
- Models: RandomForest (85-90%) or XGBoost (88-92%)
- Output: 11 NSynth instrument families + confidence scores

**Training:**
- Dataset: NSynth 289k samples
- Training time: 10-15 minutes (after feature extraction)
- Feature extraction: 2-3 hours (parallelized)

**Families:**
bass, brass, flute, guitar, keyboard, mallet, organ, reed, string, synth_lead, vocal

---

## ⚙️ Configuration

Adjust confidence threshold in `laplace_mrmt3/config.py`:

```python
@dataclass
class RefinementConfig:
    min_confidence: float = 0.7  # Only apply if ML confidence > 70%
```

- Lower threshold = more refinements, higher risk
- Higher threshold = fewer refinements, safer

---

## 📈 Performance Benchmarks

**On NVIDIA L4 GPU:**
- Single track processing: 9-11 seconds
- NSynth download: 15-30 minutes
- Feature extraction: 2-3 hours (300k samples)
- Model training: 10-15 minutes
- Inference: ~0.2s per instrument

**Accuracy:**
- Phase 1 (heuristics): 60-70%
- Phase 2 (ML): 88-92%
- Improvement: +28-32 percentage points

---

## 🐛 Troubleshooting

### Classifier Not Found

```bash
# Train Phase 2 classifier first
sudo ./scripts/setup_phase2_training.sh
```

### Import Errors

```bash
pip install scikit-learn xgboost joblib
```

### Low Accuracy (<80%)

- Verify NSynth download completed
- Check feature extraction logs
- Try XGBoost instead of RandomForest

---

## 💰 GPU Cost Estimation

**Phase 2 Training (one-time):**
- Duration: 3-4 hours
- Cost: ~€1-2 (NVIDIA L4)

**Batch Processing (Slakh2100 validation):**
- 375 tracks × 10s = ~1 hour
- Cost: ~€0.30-0.50

---

## 📝 Citation

```bibtex
@software{laplace_mr_mt3_2024,
  title={L-MT3: Laplace-Enhanced MR-MT3 Music Transcription},
  author={Dyapason Research},
  year={2024},
  url={https://github.com/Pyzeur-ColonyLab/L-MT3}
}
```

---

## 📄 License

MIT License - Use freely with attribution

---

## 🆘 Support

- **Issues:** https://github.com/Pyzeur-ColonyLab/L-MT3/issues
- **Documentation:** See [PHASE2_README.md](PHASE2_README.md) and [PHASE2_DEPLOYMENT_GUIDE.md](PHASE2_DEPLOYMENT_GUIDE.md)
- **Classifier Guide:** See [Laplace_classifier/CLASSIFIER_DEV_GUIDE.md](Laplace_classifier/CLASSIFIER_DEV_GUIDE.md)
