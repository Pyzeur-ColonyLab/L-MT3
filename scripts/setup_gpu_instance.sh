#!/bin/bash
################################################################################
# GPU Instance Automated Setup for Local MR-MT3 + Laplace Pipeline
#
# This script fully automates GPU instance setup for running the complete
# Phase 1 research pipeline with local MR-MT3 transcription.
#
# Hardware Requirements:
#   - NVIDIA GPU (T4/A2/L4) with 16GB+ VRAM
#   - 16+ CPU cores
#   - 32GB+ RAM
#   - 500GB+ storage
#
# Usage:
#   wget https://raw.githubusercontent.com/Pyzeur-ColonyLab/L-MT3/main/scripts/setup_gpu_instance.sh
#   chmod +x setup_gpu_instance.sh
#   sudo ./setup_gpu_instance.sh
#
# What it does:
#   1. Install system dependencies (CUDA, Python, git, ffmpeg)
#   2. Clone L-MT3 repository
#   3. Download babyslakh_16k dataset
#   4. Install Python dependencies (MR-MT3 + Laplace)
#   5. Download MR-MT3 pre-trained model
#   6. Run test pipeline on 10 files
#
################################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

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

log_section() {
    echo ""
    echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN} $1${NC}"
    echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    log_error "Please run as root (sudo ./setup_gpu_instance.sh)"
    exit 1
fi

# Get the actual user (not root)
ACTUAL_USER=${SUDO_USER:-$USER}
USER_HOME=$(eval echo ~$ACTUAL_USER)

log_section "GPU Instance Setup - Starting"
log_info "User: $ACTUAL_USER"
log_info "Home: $USER_HOME"
log_info "Date: $(date)"

################################################################################
# 1. System Dependencies
################################################################################

log_section "Step 1/8: Installing System Dependencies"

log_info "Updating package lists..."
apt-get update -qq

log_info "Installing base packages..."
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    git \
    wget \
    curl \
    build-essential \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    sox \
    > /dev/null

log_success "System dependencies installed"

################################################################################
# 2. CUDA Setup
################################################################################

log_section "Step 2/8: Verifying CUDA Installation"

if command -v nvidia-smi &> /dev/null; then
    log_success "NVIDIA drivers detected"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    log_warning "NVIDIA drivers not detected - GPU acceleration may not work"
fi

# Check CUDA
if [ -d "/usr/local/cuda" ]; then
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
    log_success "CUDA found at /usr/local/cuda"
else
    log_warning "CUDA not found - installing CUDA toolkit..."
    # Install CUDA 11.8 (compatible with TensorFlow 2.11)
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    dpkg -i cuda-keyring_1.0-1_all.deb
    apt-get update -qq
    apt-get -y install cuda-11-8
    export PATH="/usr/local/cuda-11.8/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}"
    log_success "CUDA 11.8 installed"
fi

################################################################################
# 3. Detect Repository Directory
################################################################################

log_section "Step 3/8: Detecting Repository Directory"

# Detect working directory from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(dirname "$SCRIPT_DIR")"

log_info "Working directory: $WORK_DIR"

if [ ! -f "$WORK_DIR/phase1_mrmt3_enhancement.py" ]; then
    log_error "Not running from L-MT3 repository! Please run from repo: sudo ./scripts/setup_gpu_instance.sh"
    exit 1
fi

log_success "Repository detected at $WORK_DIR"

################################################################################
# 4. Download Dataset (COMMENTED OUT - Manually clone and extract dataset)
################################################################################

# log_section "Step 4/8: Downloading babyslakh_16k Dataset"
#
# DATASET_DIR="$WORK_DIR/babyslakh_16k"
#
# if [ -d "$DATASET_DIR" ]; then
#     log_warning "Dataset already exists at $DATASET_DIR"
# else
#     log_info "Downloading babyslakh_16k (7GB)..."
#     cd "$WORK_DIR"
#     sudo -u $ACTUAL_USER wget -q --show-progress \
#         https://zenodo.org/record/4599666/files/babyslakh_16k.tar.gz
#
#     log_info "Extracting dataset..."
#     sudo -u $ACTUAL_USER tar -xzf babyslakh_16k.tar.gz
#     rm babyslakh_16k.tar.gz
#
#     TRACK_COUNT=$(find "$DATASET_DIR" -name "mix.wav" | wc -l)
#     log_success "Dataset extracted: $TRACK_COUNT tracks"
# fi

log_info "Skipping dataset download (assumes babyslakh_16k/ already exists)"

################################################################################
# 5. Install MR-MT3 Dependencies
################################################################################

log_section "Step 5/8: Installing MR-MT3 Dependencies"

log_info "Creating Python virtual environment..."
cd "$WORK_DIR"
sudo -u $ACTUAL_USER python3 -m venv venv
source venv/bin/activate

log_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel -q

log_info "Installing TensorFlow 2.11 with GPU support..."
pip install tensorflow==2.11.0 -q

log_info "Installing MR-MT3 requirements..."
cat > "$WORK_DIR/requirements_mrmt3_gpu.txt" << 'EOF'
# TensorFlow and JAX for GPU
tensorflow==2.11.0
jax[cuda11_cudnn82]==0.4.1
jaxlib==0.4.1+cuda11.cudnn82

# MR-MT3 Core
note-seq==0.0.5
mt3==0.1.0

# Audio Processing
librosa>=0.10.0,<0.11.0
soundfile>=0.12.0
audioread>=3.0.0

# MIDI Processing
pretty-midi>=0.2.10
mido>=1.3.0

# Scientific Computing (numpy<2.0 for compatibility)
numpy>=1.24.0,<2.0.0
scipy>=1.10.0,<1.15.0

# Compatibility
protobuf>=3.19.0,<3.20.0
tensorflow-metadata<1.10
tensorflow-datasets>=4.5.2,<4.6.0
dill>=0.3.4,<0.4.0

# Laplace Enhancement
matplotlib>=3.7.0,<4.0.0
PyYAML>=6.0
tqdm>=4.65.0
numba>=0.57.0,<0.60.0
scikit-learn>=1.3.0,<1.6.0
joblib>=1.3.0
soxr>=0.3.0
EOF

pip install -r "$WORK_DIR/requirements_mrmt3_gpu.txt"

log_success "Python dependencies installed"

################################################################################
# 6. Download MR-MT3 Model
################################################################################

log_section "Step 6/8: Downloading MR-MT3 Pre-trained Model"

MODEL_DIR="$WORK_DIR/models"
MODEL_PATH="$MODEL_DIR/mr-mt3-single-instrument.ckpt"

if [ -f "$MODEL_PATH" ]; then
    log_warning "Model already exists at $MODEL_PATH"
else
    log_info "Creating models directory..."
    mkdir -p "$MODEL_DIR"

    log_info "Downloading MR-MT3 checkpoint (400MB)..."
    cd "$MODEL_DIR"
    sudo -u $ACTUAL_USER wget -q --show-progress \
        "https://storage.googleapis.com/magentadata/models/mr-mt3/checkpoints/mr-mt3-single-instrument.ckpt"

    log_success "Model downloaded: $(du -h $MODEL_PATH | cut -f1)"
fi

################################################################################
# 7. Create Local Pipeline Script
################################################################################

log_section "Step 7/8: Creating Local MR-MT3 Pipeline Script"

PIPELINE_SCRIPT="$WORK_DIR/scripts/run_local_pipeline.sh"

cat > "$PIPELINE_SCRIPT" << 'PIPELINE_EOF'
#!/bin/bash
################################################################################
# Local MR-MT3 + Laplace Enhancement Pipeline
#
# This script runs the complete pipeline using local MR-MT3 installation
# instead of the API-based approach.
#
# Usage:
#   ./scripts/run_local_pipeline.sh [OPTIONS]
#
# Options:
#   --data-dir PATH       Input audio directory
#   --output-dir PATH     Output directory
#   --model PATH          MR-MT3 model checkpoint
#   --num-files N         Number of files to process (default: all)
#   --evaluate            Run evaluation after processing
################################################################################

set -euo pipefail

# Default configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/babyslakh_16k"
OUTPUT_DIR="$PROJECT_ROOT/pipeline_output_local"
MODEL_PATH="$PROJECT_ROOT/models/mr-mt3-single-instrument.ckpt"
NUM_FILES="all"
RUN_EVALUATION=""

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
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --num-files)
            NUM_FILES="$2"
            shift 2
            ;;
        --evaluate)
            RUN_EVALUATION="true"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Activate virtual environment
source "$PROJECT_ROOT/venv/bin/activate"

# Create output directories
mkdir -p "$OUTPUT_DIR"/{mrmt3_transcriptions,enhanced_transcriptions,metrics,logs}

echo "[INFO] Starting Local MR-MT3 + Laplace Pipeline"
echo "[INFO] Data directory: $DATA_DIR"
echo "[INFO] Output directory: $OUTPUT_DIR"
echo "[INFO] Model: $MODEL_PATH"

# Find audio files
AUDIO_FILES=($(find "$DATA_DIR" -name "mix.wav" | sort))
TOTAL_FILES=${#AUDIO_FILES[@]}

if [ "$NUM_FILES" != "all" ]; then
    AUDIO_FILES=("${AUDIO_FILES[@]:0:$NUM_FILES}")
    echo "[INFO] Processing first $NUM_FILES of $TOTAL_FILES files"
else
    echo "[INFO] Processing all $TOTAL_FILES files"
fi

# Process each file
PROCESSED=0
FAILED=0

for AUDIO_FILE in "${AUDIO_FILES[@]}"; do
    TRACK_NAME=$(basename "$(dirname "$AUDIO_FILE")")
    echo ""
    echo "[INFO] Processing: $TRACK_NAME"

    # Step 1: MR-MT3 Transcription
    echo "[INFO]   → Running MR-MT3 transcription..."
    MIDI_OUTPUT="$OUTPUT_DIR/mrmt3_transcriptions/${TRACK_NAME}.mid"

    if python3 "$PROJECT_ROOT/run_mrmt3_inference.py" \
        --model "$MODEL_PATH" \
        --audio "$AUDIO_FILE" \
        --output "$MIDI_OUTPUT" \
        --device cuda \
        2>&1 | tee -a "$OUTPUT_DIR/logs/mrmt3.log"; then
        echo "[SUCCESS]   ✓ Transcription complete"
    else
        echo "[ERROR]   ✗ Transcription failed"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Step 2: Laplace Enhancement
    echo "[INFO]   → Running Laplace enhancement..."
    ENHANCED_OUTPUT="$OUTPUT_DIR/enhanced_transcriptions/${TRACK_NAME}_enhanced.mid"

    if python3 "$PROJECT_ROOT/phase1_mrmt3_enhancement.py" \
        --midi "$MIDI_OUTPUT" \
        --audio "$AUDIO_FILE" \
        --output "$ENHANCED_OUTPUT" \
        --config "$PROJECT_ROOT/configs/enhancement.yaml" \
        2>&1 | tee -a "$OUTPUT_DIR/logs/enhancement.log"; then
        echo "[SUCCESS]   ✓ Enhancement complete"
        PROCESSED=$((PROCESSED + 1))
    else
        echo "[ERROR]   ✗ Enhancement failed"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "========================================================================"
echo "                    PIPELINE COMPLETE"
echo "========================================================================"
echo ""
echo "Total files:       ${#AUDIO_FILES[@]}"
echo "Successfully processed: $PROCESSED"
echo "Failed:            $FAILED"
echo ""
echo "Outputs:"
echo "  - MR-MT3 transcriptions: $OUTPUT_DIR/mrmt3_transcriptions/"
echo "  - Enhanced:              $OUTPUT_DIR/enhanced_transcriptions/"
echo "  - Logs:                  $OUTPUT_DIR/logs/"
echo ""

# Run evaluation if requested
if [ -n "$RUN_EVALUATION" ] && [ $PROCESSED -gt 0 ]; then
    echo "[INFO] Running evaluation..."
    python3 "$PROJECT_ROOT/evaluate_enhancement.py" \
        --baseline-dir "$OUTPUT_DIR/mrmt3_transcriptions" \
        --enhanced-dir "$OUTPUT_DIR/enhanced_transcriptions" \
        --audio-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR/evaluation" \
        2>&1 | tee "$OUTPUT_DIR/logs/evaluation.log"

    echo ""
    echo "[SUCCESS] Evaluation complete!"
    echo "[INFO] View report: $OUTPUT_DIR/evaluation/evaluation_report.html"
fi

echo "========================================================================"
PIPELINE_EOF

chmod +x "$PIPELINE_SCRIPT"
chown $ACTUAL_USER:$ACTUAL_USER "$PIPELINE_SCRIPT"

log_success "Pipeline script created"

################################################################################
# 8. Run Test Pipeline
################################################################################

log_section "Step 8/8: Running Test Pipeline (10 files)"

log_info "Starting test run..."

cd "$WORK_DIR"
source venv/bin/activate

sudo -u $ACTUAL_USER bash "$PIPELINE_SCRIPT" \
    --num-files 10 \
    --evaluate \
    2>&1 | tee "$WORK_DIR/setup_test_output.log"

log_success "Test pipeline complete!"

################################################################################
# Summary
################################################################################

log_section "Setup Complete!"

echo "Instance is ready for MR-MT3 + Laplace research pipeline"
echo ""
echo "Key Locations:"
echo "  - Repository:   $WORK_DIR"
echo "  - Dataset:      $WORK_DIR/babyslakh_16k"
echo "  - Model:        $MODEL_PATH"
echo "  - Test output:  $WORK_DIR/pipeline_output_local"
echo ""
echo "To run full dataset (233 files):"
echo "  cd $WORK_DIR"
echo "  source venv/bin/activate"
echo "  ./scripts/run_local_pipeline.sh --evaluate"
echo ""
echo "To monitor GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Setup log saved to: $WORK_DIR/setup_test_output.log"
echo ""

log_success "All done! 🎉"
