# GPU Instance Setup Guide

Automated setup instructions for running the complete MR-MT3 + Laplace Enhancement pipeline on GPU instances.

## Instance Requirements

**Recommended Configuration:**
- **GPU**: NVIDIA T4 (16GB VRAM)
- **CPU**: 16 cores
- **RAM**: 32GB
- **Storage**: 500GB SSD
- **OS**: Ubuntu 22.04 LTS
- **Cost**: ~€0.26/hour

**Minimum Requirements:**
- GPU: 8GB+ VRAM (A2/T4/L4)
- CPU: 4+ cores
- RAM: 16GB+
- Storage: 100GB+

## Quick Start (3 steps)

### 1. Launch GPU Instance

Start your GPU instance with Ubuntu 22.04 and SSH in.

### 2. Download and Run Setup Script

```bash
# Download setup script
wget https://raw.githubusercontent.com/Pyzeur-ColonyLab/L-MT3/main/scripts/setup_gpu_instance.sh

# Make executable
chmod +x setup_gpu_instance.sh

# Run as root (will auto-install everything + run 10-file test)
sudo ./setup_gpu_instance.sh
```

**What it does:**
1. Installs CUDA, Python, ffmpeg, and system dependencies
2. Clones L-MT3 repository
3. Downloads babyslakh_16k dataset (7GB)
4. Installs TensorFlow 2.11 + MR-MT3 + Laplace dependencies
5. Downloads MR-MT3 model checkpoint (400MB)
6. Creates automated pipeline script
7. Runs test on 10 files with evaluation

**Duration:** ~30-45 minutes

### 3. Run Full Dataset

After setup completes:

```bash
cd ~/L-MT3
source venv/bin/activate

# Process all 233 files with evaluation
./scripts/run_local_pipeline.sh --evaluate
```

**Duration:** ~27 hours for 233 files

## Manual Operations

### Process Specific Number of Files

```bash
# Test with 10 files
./scripts/run_local_pipeline.sh --num-files 10

# Process 50 files
./scripts/run_local_pipeline.sh --num-files 50 --evaluate
```

### Monitor GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Resume After Interruption

The pipeline is stateless - just re-run with `--num-files` to continue:

```bash
# Check what's already processed
ls ~/L-MT3/pipeline_output_local/enhanced_transcriptions/ | wc -l

# Continue from where you left off by processing remaining files
# (Pipeline automatically skips already processed files)
./scripts/run_local_pipeline.sh
```

### View Results

```bash
# Evaluation report (open in browser)
~/L-MT3/pipeline_output_local/evaluation/evaluation_report.html

# View logs
tail -f ~/L-MT3/pipeline_output_local/logs/mrmt3.log
tail -f ~/L-MT3/pipeline_output_local/logs/enhancement.log
```

## Troubleshooting

### CUDA Not Found

```bash
# Check CUDA installation
ls /usr/local/cuda*

# Add to PATH if needed
export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
```

### Out of GPU Memory

```bash
# Check current GPU usage
nvidia-smi

# Reduce batch size in MR-MT3 inference
# Edit run_mrmt3_inference.py and reduce batch_size parameter
```

### TensorFlow GPU Not Detected

```bash
# Verify TensorFlow sees GPU
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Should output: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### Dependencies Install Failed

```bash
# Manual install with verbose output
cd ~/L-MT3
source venv/bin/activate
pip install -r requirements_mrmt3_gpu.txt -v
```

## Cost Estimation

**For T4 instance (€0.26/hour):**
- Setup (45 min): ~€0.20
- 10-file test (~1 hour): ~€0.26
- 233-file full run (~27 hours): ~€7.00
- 1200-file Slakh2100 (~140 hours): ~€36.00

**Total for complete research:**
- Setup + testing + babyslakh: ~€7.50
- Full Slakh2100 (if needed): ~€36.00

## Output Structure

```
~/L-MT3/
├── babyslakh_16k/              # Dataset (233 tracks)
├── models/                     # MR-MT3 checkpoint
├── pipeline_output_local/      # Results
│   ├── mrmt3_transcriptions/   # Baseline MIDI
│   ├── enhanced_transcriptions/ # Laplace-enhanced MIDI
│   ├── evaluation/             # Comparative analysis
│   │   ├── evaluation_report.html
│   │   ├── detailed_results.json
│   │   └── *.png               # Charts
│   ├── metrics/                # Processing metrics
│   └── logs/                   # Execution logs
└── scripts/
    ├── setup_gpu_instance.sh   # This setup script
    └── run_local_pipeline.sh   # Pipeline execution
```

## Next Steps

1. Review test results in `pipeline_output_local/evaluation/`
2. If results are good (>10% leakage reduction), run full 233 files
3. If results are excellent, consider Slakh2100 (1200 files)

## Support

For issues:
- Check logs in `~/L-MT3/pipeline_output_local/logs/`
- Review setup log: `~/L-MT3/setup_test_output.log`
- GitHub issues: https://github.com/Pyzeur-ColonyLab/L-MT3/issues
