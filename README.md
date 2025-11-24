# MR-MT3 + Laplace Enhancement Pipeline

GPU-accelerated pipeline for music transcription with Laplace transform-based enhancement to reduce instrument leakage.

## Quick Start (GPU Instance)

**3-step automated setup:**

```bash
# 1. Download setup script
wget https://raw.githubusercontent.com/Pyzeur-ColonyLab/L-MT3/main/scripts/setup_gpu_instance.sh

# 2. Make executable
chmod +x setup_gpu_instance.sh

# 3. Run (auto-installs everything + tests 10 files)
sudo ./setup_gpu_instance.sh
```

**Duration:** ~45 minutes
**What it does:** Installs CUDA + TensorFlow + MR-MT3 + Laplace, downloads dataset, runs test

**After setup:**
```bash
cd ~/L-MT3
source venv/bin/activate
./scripts/run_local_pipeline.sh --evaluate  # Process all 233 files
```

📖 **Full instructions:** See [GPU_SETUP.md](GPU_SETUP.md)

---

## What This Does

**Problem:** MR-MT3 transcription has instrument leakage (notes assigned to wrong instruments)

**Solution:** Laplace transform features (decay rates, spectral analysis, attack times) to:
1. Consolidate instruments with similar acoustic signatures
2. Refine note assignments based on timbre characteristics
3. Reduce cross-instrument leakage by 10-25%

**Pipeline:**
```
Audio → MR-MT3 (local GPU) → MIDI → Laplace Enhancement → Improved MIDI
```

---

## Instance Requirements

**Recommended (T4 GPU):**
- GPU: NVIDIA T4 (16GB VRAM)
- CPU: 16 cores
- RAM: 32GB
- Storage: 500GB
- Cost: ~€0.26/hour (~€7 for 233 files)

**Minimum:**
- GPU: 8GB+ VRAM
- CPU: 4+ cores
- RAM: 16GB
- Storage: 100GB

---

## Results & Evaluation

After running the pipeline, you get:

**Evaluation Report** (`pipeline_output_local/evaluation/evaluation_report.html`):
- Visual charts comparing baseline vs enhanced
- Statistical analysis of improvement
- Per-file breakdown
- Publication-ready graphs (300 DPI)

**Example Results:**
```
Baseline:     9 instruments, high leakage
Enhanced:     7 instruments, 22.2% leakage reduction
Improvement:  Fewer cross-instrument note assignments
```

📊 **Evaluation guide:** See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)

---

## Repository Structure

```
├── GPU_SETUP.md                    # Complete GPU setup guide
├── EVALUATION_GUIDE.md             # Results analysis guide
├── LAPLACE_THEORY.md               # Mathematical background
├── scripts/
│   ├── setup_gpu_instance.sh       # Automated GPU setup
│   └── run_local_pipeline.sh       # Pipeline execution (created by setup)
├── run_mrmt3_inference.py          # MR-MT3 GPU inference wrapper
├── phase1_mrmt3_enhancement.py     # Laplace enhancement
├── evaluate_enhancement.py         # Comparative analysis
└── laplace_mrmt3/                  # Core library
    ├── feature_extraction.py       # Prony, VQT, Gammatone
    ├── consolidation.py            # Instrument merging
    ├── refinement.py               # Timbre-based adjustments
    └── metrics.py                  # Leakage evaluation
```

---

## Datasets

**babyslakh_16k** (233 tracks, ~7GB):
- Automatically downloaded by setup script
- Used for initial testing and validation

**Slakh2100** (1200 tracks, ~200GB):
- Full dataset for comprehensive evaluation
- Download manually if needed:
  ```bash
  wget https://zenodo.org/record/4599666/files/slakh2100_flac_redux.tar.gz
  tar -xzf slakh2100_flac_redux.tar.gz
  ```

---

## Manual Operations

### Process Specific Files
```bash
cd ~/L-MT3
source venv/bin/activate

# Test with 10 files
./scripts/run_local_pipeline.sh --num-files 10

# Process 50 files with evaluation
./scripts/run_local_pipeline.sh --num-files 50 --evaluate

# Full dataset (233 files, ~27 hours)
./scripts/run_local_pipeline.sh --evaluate
```

### Monitor GPU
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Check memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### View Results
```bash
# Open evaluation report in browser
xdg-open ~/L-MT3/pipeline_output_local/evaluation/evaluation_report.html

# Check logs
tail -f ~/L-MT3/pipeline_output_local/logs/mrmt3.log
tail -f ~/L-MT3/pipeline_output_local/logs/enhancement.log
```

---

## Technical Details

**Laplace Features:**
1. **Prony Analysis**: Exponential decay rates for each note
2. **Variable-Q Transform (VQT)**: Multi-resolution spectral analysis
3. **Gammatone Filterbank**: Perceptual attack/decay characteristics

**Enhancement Stages:**
1. Feature extraction from MIDI + audio
2. Decay-based onset consolidation
3. Timbre-based duration refinement
4. Program change optimization

**Metrics:**
- Cross-instrument leakage rate
- Temporal overlap ratio
- Note accuracy (F1 score with ground truth)

📖 **Mathematical details:** See [LAPLACE_THEORY.md](LAPLACE_THEORY.md)

---

## Cost Estimation (T4 GPU @ €0.26/hour)

| Task | Duration | Cost |
|------|----------|------|
| Setup + 10-file test | ~1 hour | ~€0.26 |
| babyslakh (233 files) | ~27 hours | ~€7.00 |
| Slakh2100 (1200 files) | ~140 hours | ~€36.00 |

**Total for research:** €7-43 depending on dataset size

---

## Troubleshooting

**CUDA not found:**
```bash
export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
```

**Out of GPU memory:**
- Reduce MR-MT3 batch size in `run_mrmt3_inference.py`
- Check GPU usage: `nvidia-smi`

**Dependencies failed:**
```bash
cd ~/L-MT3
source venv/bin/activate
pip install -r requirements_mrmt3_gpu.txt -v
```

📖 **Full troubleshooting:** See [GPU_SETUP.md](GPU_SETUP.md#troubleshooting)

---

## Citation

If you use this pipeline in your research:

```bibtex
@software{laplace_mr_mt3_2024,
  title={Laplace-Enhanced MR-MT3 Music Transcription},
  author={Dyapason Research},
  year={2024},
  url={https://github.com/Pyzeur-ColonyLab/L-MT3}
}
```

---

## License

MIT License - Use freely with attribution

---

## Support

- **Issues:** https://github.com/Pyzeur-ColonyLab/L-MT3/issues
- **Documentation:** See [GPU_SETUP.md](GPU_SETUP.md) and [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)
- **Theory:** See [LAPLACE_THEORY.md](LAPLACE_THEORY.md)
