#!/bin/bash
################################################################################
# MR-MT3 + Laplace Enhancement Pipeline - Production Deployment Script
#
# This script sets up the complete pipeline on internal instances:
# 1. Clone MR-MT3 repository
# 2. Install all dependencies (MR-MT3 + Laplace enhancement)
# 3. Download model checkpoint
# 4. Process audio files through MR-MT3 inference
# 5. Apply Laplace enhancement to transcriptions
# 6. Generate evaluation metrics
#
# Usage:
#   ./deploy_mrmt3_pipeline.sh [OPTIONS]
#
# Options:
#   --data-dir PATH       Input audio directory (default: ./audio_input)
#   --output-dir PATH     Output directory (default: ./pipeline_output)
#   --model-path PATH     Pre-downloaded model checkpoint path (skip download)
#   --batch-size N        MR-MT3 batch size (default: 8)
#   --gpu                 Use GPU if available (default: CPU only)
#   --workers N           Number of parallel workers (default: 4)
#   --skip-install        Skip dependency installation
#   --test-mode           Run with 1 sample only for testing
################################################################################

set -euo pipefail

# Default configuration
DATA_DIR="./audio_input"
OUTPUT_DIR="./pipeline_output"
MODEL_PATH=""
BATCH_SIZE=8
USE_GPU=false
WORKERS=4
SKIP_INSTALL=false
TEST_MODE=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gpu)
            USE_GPU=true
            shift
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --skip-install)
            SKIP_INSTALL=true
            shift
            ;;
        --test-mode)
            TEST_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

################################################################################
# 1. Environment Setup
################################################################################

log_info "Starting MR-MT3 + Laplace Enhancement Pipeline Deployment"
log_info "Project root: $PROJECT_ROOT"

# Create output directories
mkdir -p "$OUTPUT_DIR"/{mrmt3_transcriptions,enhanced_transcriptions,metrics,logs}
log_success "Created output directory structure"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
log_info "Python version: $PYTHON_VERSION"

if [[ ! "$PYTHON_VERSION" =~ ^3\.(8|9|10|11) ]]; then
    log_error "Python 3.8-3.11 required, found $PYTHON_VERSION"
    exit 1
fi

################################################################################
# 2. Install Dependencies
################################################################################

if [ "$SKIP_INSTALL" = false ]; then
    log_info "Installing dependencies..."

    # Install Laplace enhancement dependencies
    log_info "Installing Laplace enhancement dependencies..."
    pip3 install --user --upgrade \
        numpy==1.24.3 \
        scipy==1.10.1 \
        librosa==0.10.0 \
        pretty_midi==0.2.10 \
        pyyaml==6.0 \
        pytest==7.3.1 \
        2>&1 | tee "$OUTPUT_DIR/logs/laplace_install.log"

    # Check if MR-MT3 is cloned
    MRMT3_DIR="$PROJECT_ROOT/mr-mt3"
    if [ ! -d "$MRMT3_DIR" ]; then
        log_info "Cloning MR-MT3 repository..."
        cd "$PROJECT_ROOT"
        git clone https://github.com/gudgud96/MR-MT3.git mr-mt3
        cd mr-mt3
    else
        log_info "MR-MT3 already cloned at $MRMT3_DIR"
        cd "$MRMT3_DIR"
    fi

    # Install MR-MT3 dependencies
    log_info "Installing MR-MT3 dependencies..."
    if [ "$USE_GPU" = true ]; then
        pip3 install --user -r requirements-gpu.txt 2>&1 | tee "$OUTPUT_DIR/logs/mrmt3_install_gpu.log"
    else
        pip3 install --user -r requirements.txt 2>&1 | tee "$OUTPUT_DIR/logs/mrmt3_install.log"
    fi

    log_success "All dependencies installed"
else
    log_warning "Skipping dependency installation (--skip-install)"
    MRMT3_DIR="$PROJECT_ROOT/mr-mt3"
fi

################################################################################
# 3. Download Model Checkpoint
################################################################################

if [ -z "$MODEL_PATH" ]; then
    log_info "Downloading MR-MT3 model checkpoint..."

    # Use pretrained directory within MR-MT3
    MODEL_PATH="$MRMT3_DIR/pretrained/mt3.pth"

    if [ ! -f "$MODEL_PATH" ]; then
        log_info "Downloading model from HuggingFace..."
        # Use our download script
        bash "$SCRIPT_DIR/download_mrmt3_model.sh" "$MRMT3_DIR/pretrained" 2>&1 | tee "$OUTPUT_DIR/logs/model_download.log"
        log_success "Model downloaded to $MODEL_PATH"
    else
        SIZE=$(du -h "$MODEL_PATH" | cut -f1)
        log_info "Model already exists at $MODEL_PATH ($SIZE)"
    fi
else
    log_info "Using pre-downloaded model at $MODEL_PATH"
fi

################################################################################
# 4. Prepare Audio Files
################################################################################

log_info "Preparing audio files from $DATA_DIR..."

# Find all audio files
AUDIO_FILES=($(find "$DATA_DIR" -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" \)))
TOTAL_FILES=${#AUDIO_FILES[@]}

if [ $TOTAL_FILES -eq 0 ]; then
    log_error "No audio files found in $DATA_DIR"
    exit 1
fi

log_info "Found $TOTAL_FILES audio files"

# In test mode, process only 1 file
if [ "$TEST_MODE" = true ]; then
    AUDIO_FILES=("${AUDIO_FILES[0]}")
    log_warning "Test mode: processing only 1 file"
fi

################################################################################
# 5. Run MR-MT3 Inference
################################################################################

log_info "Running MR-MT3 inference on ${#AUDIO_FILES[@]} files..."

cd "$MRMT3_DIR"

TRANSCRIPTION_DIR="$OUTPUT_DIR/mrmt3_transcriptions"
INFERENCE_LOG="$OUTPUT_DIR/logs/mrmt3_inference.log"

cat > inference_config.yaml <<EOF
# MR-MT3 Inference Configuration
model_checkpoint: $MODEL_PATH
batch_size: $BATCH_SIZE
use_gpu: $USE_GPU
sample_rate: 16000
max_duration: 300  # seconds
num_workers: $WORKERS
EOF

# Create file list
FILE_LIST="$OUTPUT_DIR/audio_files.txt"
printf '%s\n' "${AUDIO_FILES[@]}" > "$FILE_LIST"

log_info "Starting MR-MT3 inference (batch size: $BATCH_SIZE)..."

# Run inference (adjust command based on actual MR-MT3 interface)
python3 -m mr_mt3.inference \
    --config inference_config.yaml \
    --file-list "$FILE_LIST" \
    --output-dir "$TRANSCRIPTION_DIR" \
    2>&1 | tee "$INFERENCE_LOG"

TRANSCRIPTION_COUNT=$(find "$TRANSCRIPTION_DIR" -name "*.mid" | wc -l)
log_success "MR-MT3 inference complete: $TRANSCRIPTION_COUNT transcriptions"

################################################################################
# 6. Run Laplace Enhancement
################################################################################

log_info "Running Laplace enhancement on transcriptions..."

cd "$PROJECT_ROOT"

ENHANCEMENT_DIR="$OUTPUT_DIR/enhanced_transcriptions"
ENHANCEMENT_LOG="$OUTPUT_DIR/logs/laplace_enhancement.log"

# Process each transcription
PROCESSED=0
FAILED=0

for AUDIO_FILE in "${AUDIO_FILES[@]}"; do
    BASENAME=$(basename "$AUDIO_FILE" | sed 's/\.[^.]*$//')
    TRANSCRIPTION="$TRANSCRIPTION_DIR/${BASENAME}.mid"
    ENHANCED="$ENHANCEMENT_DIR/${BASENAME}_enhanced.mid"

    if [ ! -f "$TRANSCRIPTION" ]; then
        log_warning "Transcription not found: $TRANSCRIPTION (skipping)"
        continue
    fi

    log_info "Enhancing: $BASENAME"

    # Run enhancement pipeline
    python3 phase1_mrmt3_enhancement.py \
        --midi "$TRANSCRIPTION" \
        --audio "$AUDIO_FILE" \
        --output "$ENHANCED" \
        --config configs/enhancement.yaml \
        2>&1 | tee -a "$ENHANCEMENT_LOG"

    if [ $? -eq 0 ]; then
        ((PROCESSED++))
        log_success "Enhanced: $BASENAME"
    else
        ((FAILED++))
        log_error "Failed: $BASENAME"
    fi
done

log_success "Laplace enhancement complete: $PROCESSED succeeded, $FAILED failed"

################################################################################
# 7. Generate Metrics Report
################################################################################

log_info "Generating evaluation metrics..."

METRICS_REPORT="$OUTPUT_DIR/metrics/evaluation_report.json"

python3 <<EOF
import json
import sys
from pathlib import Path

# Collect metrics from enhancement logs
metrics = {
    "pipeline_version": "Phase1_MR-MT3_Laplace",
    "total_files": ${#AUDIO_FILES[@]},
    "mrmt3_transcriptions": $TRANSCRIPTION_COUNT,
    "enhanced_transcriptions": $PROCESSED,
    "failed_enhancements": $FAILED,
    "processing_stats": {
        "batch_size": $BATCH_SIZE,
        "workers": $WORKERS,
        "use_gpu": $USE_GPU
    }
}

with open("$METRICS_REPORT", 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"Metrics saved to $METRICS_REPORT")
EOF

log_success "Metrics report generated"

################################################################################
# 8. Summary
################################################################################

echo ""
echo "========================================================================"
echo "                    PIPELINE DEPLOYMENT COMPLETE"
echo "========================================================================"
echo ""
echo "Input:              $DATA_DIR (${#AUDIO_FILES[@]} files)"
echo "Output:             $OUTPUT_DIR"
echo "MR-MT3 transcripts: $TRANSCRIPTION_COUNT"
echo "Enhanced:           $PROCESSED"
echo "Failed:             $FAILED"
echo ""
echo "Outputs:"
echo "  - MR-MT3 transcriptions:   $OUTPUT_DIR/mrmt3_transcriptions/"
echo "  - Enhanced transcriptions: $OUTPUT_DIR/enhanced_transcriptions/"
echo "  - Metrics:                 $OUTPUT_DIR/metrics/"
echo "  - Logs:                    $OUTPUT_DIR/logs/"
echo ""
echo "========================================================================"

log_success "Deployment complete!"
