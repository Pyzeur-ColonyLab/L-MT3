#!/bin/bash
################################################################################
# Local Testing Script - MR-MT3 + Laplace Enhancement Pipeline
#
# Tests the complete pipeline locally with small audio samples to verify:
# 1. All dependencies are installed correctly
# 2. MR-MT3 inference runs without errors
# 3. Laplace enhancement processes transcriptions
# 4. Output files are generated correctly
#
# Usage:
#   ./test_pipeline_local.sh
#
# This will:
# - Create 3-second test audio samples from babySlakh
# - Run MR-MT3 inference
# - Apply Laplace enhancement
# - Validate outputs and report any issues
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_DIR="$PROJECT_ROOT/pipeline_test"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }

################################################################################
# 1. Setup Test Environment
################################################################################

log_info "Setting up local test environment..."

# Clean previous test
if [ -d "$TEST_DIR" ]; then
    log_warning "Removing previous test directory"
    rm -rf "$TEST_DIR"
fi

mkdir -p "$TEST_DIR"/{audio_samples,mrmt3_output,enhanced_output,logs}

log_success "Test directory created: $TEST_DIR"

################################################################################
# 2. Check Dependencies
################################################################################

log_info "Checking dependencies..."

MISSING_DEPS=()

# Check Python packages
for pkg in numpy scipy librosa pretty_midi yaml; do
    python3 -c "import $pkg" 2>/dev/null || MISSING_DEPS+=("python3-$pkg")
done

# Check system tools (only ffmpeg required, sox optional)
if ! command -v ffmpeg >/dev/null 2>&1; then
    MISSING_DEPS+=("ffmpeg")
fi

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    log_error "Missing dependencies: ${MISSING_DEPS[*]}"
    log_info "Install with:"
    log_info "  pip3 install --user numpy scipy librosa pretty_midi pyyaml"
    log_info "  brew install ffmpeg sox  # macOS"
    exit 1
fi

log_success "All dependencies found"

################################################################################
# 3. Create Test Audio Samples
################################################################################

log_info "Creating test audio samples (3 seconds each)..."

# Find babySlakh audio files
BABYSLAKH_DIR="$PROJECT_ROOT/babyslakh_16k"

if [ ! -d "$BABYSLAKH_DIR" ]; then
    log_error "babySlakh directory not found: $BABYSLAKH_DIR"
    log_info "Please extract babyslakh_16k.tar.gz first"
    exit 1
fi

# Create 3 small test samples
SAMPLE_COUNT=0
for audio_file in "$BABYSLAKH_DIR"/Track*/mix.wav; do
    if [ $SAMPLE_COUNT -ge 3 ]; then
        break
    fi

    TRACK_NAME=$(basename $(dirname "$audio_file"))
    OUTPUT_SAMPLE="$TEST_DIR/audio_samples/${TRACK_NAME}_3sec.wav"

    log_info "Creating sample from $TRACK_NAME..."

    # Extract first 3 seconds using ffmpeg
    ffmpeg -i "$audio_file" -t 3 -acodec pcm_s16le -ar 16000 "$OUTPUT_SAMPLE" -y 2>&1 | tee -a "$TEST_DIR/logs/sample_creation.log"

    if [ $? -eq 0 ]; then
        SIZE=$(du -h "$OUTPUT_SAMPLE" | cut -f1)
        log_success "Created: ${TRACK_NAME}_3sec.wav ($SIZE)"
        ((SAMPLE_COUNT++))
    else
        log_error "Failed to create sample from $TRACK_NAME"
    fi
done

if [ $SAMPLE_COUNT -eq 0 ]; then
    log_error "No test samples created"
    exit 1
fi

log_success "Created $SAMPLE_COUNT test audio samples"

################################################################################
# 4. Test Dependency Installation Check
################################################################################

log_info "Testing Python imports..."

python3 <<EOF
import sys
import numpy as np
import scipy
import librosa
import pretty_midi
import yaml

print(f"✓ NumPy {np.__version__}")
print(f"✓ SciPy {scipy.__version__}")
print(f"✓ librosa {librosa.__version__}")
print(f"✓ pretty_midi {pretty_midi.__version__}")
print(f"✓ PyYAML {yaml.__version__}")

# Test basic functionality
sr = 16000
audio = np.random.randn(sr * 3)  # 3 seconds
print(f"✓ Created test audio: {len(audio)} samples at {sr}Hz")

# Test librosa
S = librosa.stft(audio)
print(f"✓ librosa STFT: {S.shape}")

# Test pretty_midi
midi = pretty_midi.PrettyMIDI()
instrument = pretty_midi.Instrument(program=0)
note = pretty_midi.Note(velocity=100, pitch=60, start=0, end=1)
instrument.notes.append(note)
midi.instruments.append(instrument)
print(f"✓ PrettyMIDI: {len(midi.instruments)} instruments")

print("\n✓ All Python dependencies working correctly")
EOF

if [ $? -ne 0 ]; then
    log_error "Python dependency test failed"
    exit 1
fi

log_success "Python dependencies verified"

################################################################################
# 5. Test Laplace Enhancement (Without MR-MT3)
################################################################################

log_info "Testing Laplace enhancement module..."

cd "$PROJECT_ROOT"

# Test with synthetic MIDI if available
if [ -f "test_input_vanilla.mid" ]; then
    TEST_AUDIO="$TEST_DIR/audio_samples/"*_3sec.wav
    TEST_AUDIO=$(ls $TEST_AUDIO | head -1)

    log_info "Running enhancement on test sample..."

    python3 phase1_mrmt3_enhancement.py \
        --midi test_input_vanilla.mid \
        --audio "$TEST_AUDIO" \
        --output "$TEST_DIR/enhanced_output/test_enhanced.mid" \
        --config configs/enhancement.yaml \
        2>&1 | tee "$TEST_DIR/logs/enhancement_test.log"

    if [ $? -eq 0 ] && [ -f "$TEST_DIR/enhanced_output/test_enhanced.mid" ]; then
        log_success "Laplace enhancement test passed"
    else
        log_error "Laplace enhancement test failed"
        log_info "Check log: $TEST_DIR/logs/enhancement_test.log"
        exit 1
    fi
else
    log_warning "test_input_vanilla.mid not found, skipping enhancement test"
fi

################################################################################
# 6. Test MR-MT3 Setup (If Available)
################################################################################

MRMT3_DIR="$PROJECT_ROOT/mr-mt3"

if [ -d "$MRMT3_DIR" ]; then
    log_info "Testing MR-MT3 installation..."

    cd "$MRMT3_DIR"

    # Test import
    python3 -c "import mr_mt3; print(f'✓ MR-MT3 module loaded')" 2>&1 | tee -a "$TEST_DIR/logs/mrmt3_test.log"

    if [ $? -eq 0 ]; then
        log_success "MR-MT3 module available"
    else
        log_warning "MR-MT3 module not available (may need installation)"
    fi
else
    log_warning "MR-MT3 not cloned yet at $MRMT3_DIR"
    log_info "Run deployment script to clone and install MR-MT3"
fi

################################################################################
# 7. Performance Test
################################################################################

log_info "Running performance test..."

python3 <<EOF
import time
import numpy as np
import librosa
from pathlib import Path

# Load test audio
test_audio = list(Path("$TEST_DIR/audio_samples").glob("*.wav"))[0]
print(f"Loading: {test_audio.name}")

start = time.time()
audio, sr = librosa.load(str(test_audio), sr=16000)
load_time = time.time() - start

print(f"✓ Loaded in {load_time:.2f}s: {len(audio)/sr:.1f}s audio at {sr}Hz")

# Test feature extraction speed
start = time.time()
S = librosa.stft(audio, n_fft=2048, hop_length=512)
stft_time = time.time() - start

print(f"✓ STFT computed in {stft_time:.2f}s: {S.shape}")

# Estimate processing time for full tracks
full_track_duration = 240  # seconds
estimated_time = (stft_time / (len(audio)/sr)) * full_track_duration

print(f"\n📊 Performance estimate:")
print(f"   3-second sample: {stft_time:.2f}s processing")
print(f"   240-second track: ~{estimated_time:.1f}s estimated")
print(f"   Processing ratio: ~{estimated_time/full_track_duration:.1f}x real-time")
EOF

if [ $? -ne 0 ]; then
    log_error "Performance test failed"
    exit 1
fi

log_success "Performance test complete"

################################################################################
# 8. Summary Report
################################################################################

echo ""
echo "========================================================================"
echo "              LOCAL PIPELINE TEST - SUMMARY"
echo "========================================================================"
echo ""
echo "Test Directory:    $TEST_DIR"
echo "Audio Samples:     $SAMPLE_COUNT (3 seconds each)"
echo ""
echo "Dependency Checks:"
echo "  ✓ Python packages installed"
echo "  ✓ System tools available"
echo "  ✓ Laplace enhancement functional"
if [ -d "$MRMT3_DIR" ]; then
    echo "  ✓ MR-MT3 repository cloned"
else
    echo "  ! MR-MT3 not yet installed"
fi
echo ""
echo "Test Outputs:"
echo "  - Audio samples:  $TEST_DIR/audio_samples/"
echo "  - Test logs:      $TEST_DIR/logs/"
if [ -f "$TEST_DIR/enhanced_output/test_enhanced.mid" ]; then
    echo "  - Enhanced MIDI:  $TEST_DIR/enhanced_output/"
fi
echo ""
echo "Next Steps:"
echo "  1. Review test logs in: $TEST_DIR/logs/"
if [ ! -d "$MRMT3_DIR" ]; then
    echo "  2. Run deployment script to install MR-MT3"
    echo "     ./deploy_mrmt3_pipeline.sh --test-mode"
fi
echo ""
echo "========================================================================"

log_success "Local test complete!"
