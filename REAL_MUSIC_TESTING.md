# Real Music Testing Guide

Comprehensive guide for testing the trained L-MT3 Laplace Classifier on real music files.

---

## 🎯 Quick Start

### Option A: Using Helper Script (Recommended)

```bash
# 1. Place your music files in a directory
mkdir music_samples
cp ~/Downloads/*.wav music_samples/

# 2. Run testing
./run_real_music_test.sh music_samples

# 3. View results
cat test_results/summary_report.txt
```

### Option B: Using Python Script Directly

```bash
# Process directory
python3 test_real_music.py \
    --audio-dir ./music_samples \
    --output-dir ./test_results \
    --classifier ./laplace_classifier.pkl \
    --model ./models/mr_mt3.pth

# Process single file
python3 test_real_music.py \
    --audio ./song.wav \
    --output-dir ./test_results \
    --classifier ./laplace_classifier.pkl \
    --model ./models/mr_mt3.pth
```

---

## 📁 Output Structure

After processing, you'll get a well-organized directory structure:

```
test_results/
├── summary_report.json          # Machine-readable batch summary
├── summary_report.txt           # Human-readable batch summary
│
├── song_01/                     # Per-track results
│   ├── song_01_baseline.mid     # MR-MT3 baseline transcription
│   ├── song_01_enhanced.mid     # ML-enhanced transcription
│   ├── song_01_report.json      # Detailed track report (JSON)
│   ├── song_01_report.txt       # Detailed track report (text)
│   └── instruments/             # Per-instrument MIDI files
│       ├── song_01_00_acoustic_grand_piano_p0.mid
│       ├── song_01_01_acoustic_bass_p32.mid
│       ├── song_01_02_violin_p40.mid
│       └── ...
│
├── song_02/
│   └── ...
│
└── song_03/
    └── ...
```

---

## 📊 Understanding Reports

### Summary Report (`summary_report.txt`)

High-level statistics across all processed tracks:

```
L-MT3 LAPLACE CLASSIFIER - BATCH TESTING SUMMARY
================================================================================

Total tracks processed: 10
Successful: 10
Failed: 0
Success rate: 100.0%

================================================================================
PROCESSING TIME
================================================================================
Total: 156.34s
Average per track: 15.63s

================================================================================
AVERAGE STATISTICS
================================================================================
Baseline instruments: 12.4
Enhanced instruments: 8.1
Instrument reduction: 4.3

================================================================================
INDIVIDUAL TRACKS
================================================================================

✓ song_01
   Enhanced MIDI: test_results/song_01/song_01_enhanced.mid

✓ song_02
   Enhanced MIDI: test_results/song_02/song_02_enhanced.mid
...
```

### Track Report (`song_01_report.txt`)

Detailed analysis for each track:

```
L-MT3 LAPLACE CLASSIFIER TEST REPORT
================================================================================

Track: song_01
Timestamp: 2025-12-11T10:30:45
Status: SUCCESS

================================================================================
INPUT
================================================================================
Audio file: /path/to/song_01.wav
Duration: 180.45s

================================================================================
BASELINE TRANSCRIPTION (MR-MT3)
================================================================================
Instruments: 14
Total notes: 3456
MIDI file: test_results/song_01/song_01_baseline.mid

================================================================================
ML-ENHANCED TRANSCRIPTION (Phase 2)
================================================================================
Instruments: 9
Total notes: 3421
MIDI file: test_results/song_01/song_01_enhanced.mid

Improvement:
  Instrument reduction: 5 (35.7%)

ML Refinement:
  Method: ml_classifier
  Details:
    Refinements applied: 8
    Average confidence: 87.2%
    By family:
      keyboard: 3
      string: 2
      brass: 2
      bass: 1

================================================================================
PER-INSTRUMENT MIDI FILES
================================================================================
Exported: 9 files
  - song_01_00_acoustic_grand_piano_p0.mid
  - song_01_01_acoustic_bass_p32.mid
  - song_01_02_violin_p40.mid
  - song_01_03_trumpet_p56.mid
  ...

================================================================================
PERFORMANCE
================================================================================
Feature extraction: 2.34s
Consolidation: 0.89s
Refinement: 1.12s
Total: 4.35s

Overall processing time: 15.67s

================================================================================
FILES GENERATED
================================================================================
baseline_midi: test_results/song_01/song_01_baseline.mid
enhanced_midi: test_results/song_01/song_01_enhanced.mid
instruments: [...list of instrument files...]
report: test_results/song_01/song_01_report.json
text_report: test_results/song_01/song_01_report.txt
```

---

## 🔧 Advanced Usage

### Skip MR-MT3 Inference (If You Already Have Baseline MIDI)

If you already have baseline MIDI files from MR-MT3:

```bash
python3 test_real_music.py \
    --audio ./song.wav \
    --baseline-midi ./song_baseline.mid \
    --output-dir ./test_results \
    --classifier ./laplace_classifier.pkl
```

### Custom Configuration

Create a custom enhancement configuration:

```yaml
# custom_config.yaml
refinement:
  min_confidence: 0.8  # Higher threshold = safer refinements

consolidation:
  strategy: conservative  # conservative, balanced, or aggressive
```

Use it:

```bash
python3 test_real_music.py \
    --audio-dir ./music_samples \
    --output-dir ./test_results \
    --classifier ./laplace_classifier.pkl \
    --config ./custom_config.yaml
```

### Verbose Mode

For debugging or detailed progress:

```bash
python3 test_real_music.py \
    --audio-dir ./music_samples \
    --output-dir ./test_results \
    --classifier ./laplace_classifier.pkl \
    --verbose
```

---

## 📝 Per-Instrument MIDI File Naming

Files are named with a descriptive pattern:

```
{track_name}_{index:02d}_{instrument_name}_p{program_number}.mid
```

**Examples:**
- `song_01_00_acoustic_grand_piano_p0.mid`
- `song_01_01_acoustic_bass_p32.mid`
- `song_01_02_violin_p40.mid`
- `song_01_03_trumpet_p56.mid`

**Pattern breakdown:**
- `song_01`: Original track name
- `00`: Instrument index (order in MIDI)
- `acoustic_grand_piano`: Human-readable instrument name
- `p0`: MIDI program number (General MIDI standard)

This makes it easy to:
- Identify instruments at a glance
- Sort files by instrument type
- Import into DAWs with proper naming

---

## 🎵 Supported Audio Formats

The pipeline supports all formats readable by `librosa`:
- `.wav` (recommended for best quality)
- `.mp3`
- `.flac`
- `.ogg`
- `.m4a`

**Recommendation:** Use `.wav` or `.flac` for best transcription quality.

---

## 🚀 GPU Instance Workflow

### Step 1: Prepare Instance

```bash
# SSH to GPU instance
ssh ubuntu@your-instance-ip

# Navigate to L-MT3
cd ~/L-MT3

# Pull latest code
git pull

# Activate environment
source venv/bin/activate
```

### Step 2: Upload Music Files

From your local machine:

```bash
# Upload music directory
scp -r ./my_music_samples ubuntu@your-instance-ip:~/L-MT3/music_samples
```

Or use rsync for better performance:

```bash
rsync -avz --progress ./my_music_samples ubuntu@your-instance-ip:~/L-MT3/
```

### Step 3: Run Testing

On the GPU instance:

```bash
# Make script executable
chmod +x run_real_music_test.sh

# Run testing
./run_real_music_test.sh music_samples
```

### Step 4: Download Results

From your local machine:

```bash
# Download results
scp -r ubuntu@your-instance-ip:~/L-MT3/test_results ./

# Or with rsync
rsync -avz --progress ubuntu@your-instance-ip:~/L-MT3/test_results ./
```

---

## 🐛 Troubleshooting

### Issue: "Classifier not found"

**Solution:**
```bash
# Train classifier first
sudo ./scripts/setup_phase2_training.sh
```

### Issue: "CUDA out of memory"

**Solutions:**
1. Process files one at a time instead of batch:
   ```bash
   for file in music_samples/*.wav; do
       python3 test_real_music.py --audio "$file" --output-dir ./test_results --classifier ./laplace_classifier.pkl --model ./models/mr_mt3.pth
   done
   ```

2. Use smaller audio files (split long tracks)

3. Upgrade to instance with more VRAM (16GB recommended)

### Issue: "No audio files found"

**Check:**
- Audio files are in the specified directory
- File extensions are supported (.wav, .mp3, .flac, .ogg, .m4a)
- Directory path is correct

### Issue: "MR-MT3 inference failed"

**Check:**
1. Model path is correct
2. GPU is available: `nvidia-smi`
3. PyTorch CUDA is installed: `python3 -c "import torch; print(torch.cuda.is_available())"`

---

## 📊 Expected Processing Times

On NVIDIA L4 GPU:
- MR-MT3 inference: ~8-10s per track
- ML enhancement: ~1-2s per track
- Per-instrument export: <1s per track
- **Total: ~10-15s per track**

For a batch of 10 tracks: ~2-3 minutes

---

## 💡 Best Practices

### 1. Organize Your Music Library

```bash
music_samples/
├── rock/
│   ├── song1.wav
│   └── song2.wav
├── jazz/
│   ├── song3.wav
│   └── song4.wav
└── classical/
    ├── song5.wav
    └── song6.wav
```

Process by genre:
```bash
./run_real_music_test.sh music_samples/rock results_rock
./run_real_music_test.sh music_samples/jazz results_jazz
./run_real_music_test.sh music_samples/classical results_classical
```

### 2. Use Descriptive Names

Name your audio files descriptively:
- ✅ `beethoven_symphony_5_mvt1.wav`
- ❌ `audio_001.wav`

This makes reports much easier to read.

### 3. Keep Baseline MIDIs

Save baseline MIDI files for later comparison:
```bash
# Copy baselines to separate directory
mkdir baselines
cp test_results/*/song_*_baseline.mid baselines/
```

### 4. Version Your Results

```bash
# Add date to results directory
./run_real_music_test.sh music_samples results_$(date +%Y%m%d)
```

---

## 🔬 Analyzing Results

### Compare Baseline vs Enhanced

Use a MIDI viewer or DAW to compare:
1. Import both `_baseline.mid` and `_enhanced.mid`
2. Compare instrument assignments
3. Check for leakage reduction

### Evaluate Per-Instrument Files

- Import individual instrument files into your DAW
- Verify instrument classification accuracy
- Check for cross-instrument leakage

### Statistical Analysis

Use the JSON reports for programmatic analysis:

```python
import json

# Load summary
with open('test_results/summary_report.json') as f:
    summary = json.load(f)

# Average instrument reduction
avg_reduction = summary['averages']['instrument_reduction']
print(f"Average instrument reduction: {avg_reduction:.1f}")

# Success rate
success_rate = summary['successful_tracks'] / summary['total_tracks']
print(f"Success rate: {success_rate:.1%}")
```

---

## 📚 References

- **Main README:** [README.md](README.md)
- **Phase 2 Guide:** [PHASE2_README.md](PHASE2_README.md)
- **Deployment Guide:** [PHASE2_DEPLOYMENT_GUIDE.md](PHASE2_DEPLOYMENT_GUIDE.md)
- **Classifier Development:** [Laplace_classifier/CLASSIFIER_DEV_GUIDE.md](Laplace_classifier/CLASSIFIER_DEV_GUIDE.md)

---

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review error messages in verbose mode: `--verbose`
3. Check track-specific reports: `test_results/<track>/<track>_report.txt`
4. Open an issue: https://github.com/Pyzeur-ColonyLab/L-MT3/issues

---

## 🎉 Example Output

After processing, you'll have everything you need:

✅ **Enhanced MIDI files** with improved instrument classification
✅ **Per-instrument MIDI files** ready for DAW import
✅ **Detailed reports** showing ML classifier decisions
✅ **Summary statistics** across your entire music library
✅ **Organized directory structure** for easy navigation

Happy transcribing! 🎵
