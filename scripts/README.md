# MR-MT3 + Laplace Enhancement - Deployment Scripts

Complete pipeline for running MR-MT3 music transcription with Laplace transform-based post-processing enhancement.

## 📁 Files

### 1. `test_pipeline_local.sh`
**Purpose**: Local testing with small samples to verify dependencies and functionality

**What it does**:
- Creates 3-second audio samples from babySlakh
- Tests all Python dependencies (numpy, scipy, librosa, pretty_midi, pyyaml)
- Runs Laplace enhancement on test samples
- Validates output files are generated correctly
- Reports performance metrics

**Usage**:
```bash
./scripts/test_pipeline_local.sh
```

**Expected Output**:
```
✓ All dependencies found
✓ Created 3 test audio samples
✓ Python dependencies verified
✓ Laplace enhancement test passed
✓ Performance test complete
```

**Test Directory**: `research/pipeline_test/`
- `audio_samples/` - 3-second test clips
- `enhanced_output/` - Enhanced MIDI files
- `logs/` - Detailed logs for debugging

---

### 2. `deploy_mrmt3_pipeline.sh`
**Purpose**: Production deployment on internal instances for processing full audio datasets

**What it does**:
1. Clones MR-MT3 repository
2. Installs all dependencies (MR-MT3 + Laplace enhancement)
3. Downloads model checkpoint
4. Runs MR-MT3 inference on audio files
5. Applies Laplace enhancement to transcriptions
6. Generates evaluation metrics

**Usage**:

#### Basic Usage (Process all audio in a directory):
```bash
./scripts/deploy_mrmt3_pipeline.sh \
    --data-dir /path/to/audio/files \
    --output-dir /path/to/output
```

#### Test Mode (Process 1 file only):
```bash
./scripts/deploy_mrmt3_pipeline.sh \
    --data-dir /path/to/audio/files \
    --output-dir /path/to/output \
    --test-mode
```

#### GPU-Enabled Processing:
```bash
./scripts/deploy_mrmt3_pipeline.sh \
    --data-dir /path/to/audio/files \
    --output-dir /path/to/output \
    --gpu \
    --batch-size 16
```

#### Skip Installation (Re-run on existing setup):
```bash
./scripts/deploy_mrmt3_pipeline.sh \
    --data-dir /path/to/audio/files \
    --output-dir /path/to/output \
    --skip-install
```

**All Options**:
```
--data-dir PATH       Input audio directory (default: ./audio_input)
--output-dir PATH     Output directory (default: ./pipeline_output)
--model-path PATH     Pre-downloaded model checkpoint path
--batch-size N        MR-MT3 batch size (default: 8)
--gpu                 Use GPU if available
--workers N           Parallel workers (default: 4)
--skip-install        Skip dependency installation
--test-mode           Process only 1 file for testing
```

**Output Structure**:
```
pipeline_output/
├── mrmt3_transcriptions/     # Raw MR-MT3 outputs
├── enhanced_transcriptions/  # Laplace-enhanced outputs
├── metrics/                  # Evaluation reports
└── logs/                     # Processing logs
```

---

## 🚀 Quick Start

### Step 1: Local Testing
```bash
# Test locally with small samples
cd /Volumes/T7/Dyapason/Fourier_Laplace/Laplace/research
./scripts/test_pipeline_local.sh
```

**Expected time**: ~2 minutes
**Purpose**: Verify all dependencies work correctly

### Step 2: Deploy on Internal Instance

```bash
# SSH to your internal instance
ssh your-instance

# Clone repository
git clone <your-repo-url>
cd Laplace/research

# Run deployment (test mode first)
./scripts/deploy_mrmt3_pipeline.sh \
    --data-dir ./audio_input \
    --output-dir ./pipeline_output \
    --test-mode

# If test succeeds, run full processing
./scripts/deploy_mrmt3_pipeline.sh \
    --data-dir ./audio_input \
    --output-dir ./pipeline_output \
    --batch-size 16 \
    --workers 8
```

---

## 📋 Dependencies

### Python Packages (Laplace Enhancement)
```
numpy==1.24.3
scipy==1.10.1
librosa==0.10.0
pretty_midi==0.2.10
pyyaml==6.0
pytest==7.3.1
```

### MR-MT3 Dependencies
Installed automatically by deployment script from `mr-mt3/requirements.txt`

### System Tools
- **ffmpeg** - Audio file processing (required)
- **Python 3.8-3.11** - Runtime environment

**Install on macOS**:
```bash
brew install ffmpeg python@3.10
```

**Install on Linux**:
```bash
sudo apt-get update
sudo apt-get install ffmpeg python3.10 python3-pip
```

---

## 🔧 Troubleshooting

### Issue: Missing Python packages
```bash
# Install manually
pip3 install --user numpy scipy librosa pretty_midi pyyaml
```

### Issue: FFmpeg not found
```bash
# macOS
brew install ffmpeg

# Linux
sudo apt-get install ffmpeg
```

### Issue: MR-MT3 clone fails
```bash
# Clone manually
cd /path/to/research
git clone https://github.com/sony/mr-mt3.git
cd mr-mt3
pip3 install --user -r requirements.txt
```

### Issue: Out of memory during processing
```bash
# Reduce batch size
./scripts/deploy_mrmt3_pipeline.sh \
    --batch-size 4 \
    --workers 2
```

### Issue: GPU not detected
```bash
# Check GPU availability
python3 -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA-enabled PyTorch
pip3 install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 📊 Expected Processing Times

### Local Test (3-second samples)
- Audio sample creation: ~1s per file
- Feature extraction: ~0.5s per file
- Enhancement: ~0.3s per file
- **Total**: ~2-3 minutes for 3 samples

### Production (240-second tracks)
- MR-MT3 inference: ~30-60s per track (GPU) / ~180-300s (CPU)
- Laplace enhancement: ~160-180s per track
- **Total**: ~3-5 minutes per track (GPU) / ~6-8 minutes (CPU)

### Full babySlakh (20 tracks)
- **Estimated**: 1-2 hours (GPU) / 2-3 hours (CPU)

---

## 🎯 Validation

### Test Script Success Indicators:
```
✓ All dependencies found
✓ Created 3 test audio samples
✓ Python dependencies verified
✓ Laplace enhancement test passed
```

### Deployment Script Success Indicators:
```
[SUCCESS] Model downloaded
[SUCCESS] MR-MT3 inference complete: N transcriptions
[SUCCESS] Laplace enhancement complete: N succeeded, 0 failed
[SUCCESS] Deployment complete!
```

### Output File Checks:
```bash
# Verify outputs exist
ls pipeline_output/mrmt3_transcriptions/*.mid
ls pipeline_output/enhanced_transcriptions/*_enhanced.mid
ls pipeline_output/metrics/evaluation_report.json
```

---

## 📈 Performance Optimization

### For CPU-Only Instances:
```bash
./scripts/deploy_mrmt3_pipeline.sh \
    --batch-size 4 \
    --workers 4
```

### For GPU Instances:
```bash
./scripts/deploy_mrmt3_pipeline.sh \
    --gpu \
    --batch-size 16 \
    --workers 8
```

### For Large Datasets (>100 files):
```bash
# Process in chunks
find audio_input -name "*.wav" | split -l 50 - chunk_

# Process each chunk
for chunk in chunk_*; do
    ./scripts/deploy_mrmt3_pipeline.sh \
        --file-list $chunk \
        --output-dir output_chunk_${chunk}
done
```

---

## 🔍 Monitoring

### Check Progress:
```bash
# Watch log files
tail -f pipeline_output/logs/mrmt3_inference.log
tail -f pipeline_output/logs/laplace_enhancement.log
```

### Check Resource Usage:
```bash
# CPU/Memory
htop

# GPU
nvidia-smi -l 1
```

---

## 📞 Support

### Logs to Check:
1. `pipeline_test/logs/` - Local test logs
2. `pipeline_output/logs/mrmt3_install.log` - MR-MT3 installation
3. `pipeline_output/logs/mrmt3_inference.log` - MR-MT3 processing
4. `pipeline_output/logs/laplace_enhancement.log` - Enhancement processing

### Common Log Locations:
```bash
# Test script logs
cat pipeline_test/logs/sample_creation.log
cat pipeline_test/logs/enhancement_test.log

# Deployment logs
cat pipeline_output/logs/model_download.log
cat pipeline_output/logs/mrmt3_inference.log
cat pipeline_output/logs/laplace_enhancement.log
```

---

## 📝 Notes

### Model Checkpoint
The MR-MT3 model checkpoint URL in the deployment script (`deploy_mrmt3_pipeline.sh:167`) needs to be updated with the actual release URL when available. Check the [MR-MT3 repository](https://github.com/sony/mr-mt3) for official model releases.

### Audio Format Support
- **Recommended**: WAV files (16 kHz, mono)
- **Supported**: MP3, FLAC (automatically converted)
- **Note**: All audio is resampled to 16 kHz for MR-MT3 compatibility

### Configuration
Laplace enhancement uses configuration from `configs/enhancement.yaml`:
- Conservative strategy (default): High precision, low recall
- Balanced strategy: Moderate precision/recall trade-off
- Aggressive strategy: Lower precision, high recall

Edit `configs/enhancement.yaml` to adjust thresholds.
