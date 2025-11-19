#!/bin/bash
################################################################################
# API-Based MR-MT3 + Laplace Enhancement Pipeline
#
# This script uses the deployed ml-api.dyapason.io service instead of local
# MR-MT3 installation, eliminating dependency management complexity.
#
# Workflow:
# 1. Upload audio files to API → receive job IDs
# 2. Trigger transcription via API
# 3. Poll for completion status
# 4. Download MIDI transcriptions
# 5. Apply Laplace enhancement
# 6. Generate evaluation metrics
#
# Usage:
#   ./deploy_api_pipeline.sh [OPTIONS]
#
# Options:
#   --data-dir PATH       Input audio directory (default: ./audio_input)
#   --output-dir PATH     Output directory (default: ./pipeline_output)
#   --api-url URL         API base URL (default: https://ml-api.dyapason.io)
#   --confidence FLOAT    Confidence threshold 0.0-1.0 (default: 0.1)
#   --batch-size N        API batch size 1-32 (default: 8)
#   --test-mode           Process only 1 file for testing
#   --evaluate            Run comparative evaluation after processing
################################################################################

set -euo pipefail

# Default configuration
DATA_DIR="./audio_input"
OUTPUT_DIR="./pipeline_output"
API_URL="https://ml-api.dyapason.io"
CONFIDENCE=0.1
BATCH_SIZE=8
TEST_MODE=false
RUN_EVALUATION=""
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
        --api-url)
            API_URL="$2"
            shift 2
            ;;
        --confidence)
            CONFIDENCE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --test-mode)
            TEST_MODE=true
            shift
            ;;
        --evaluate)
            RUN_EVALUATION=true
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
CYAN='\033[0;36m'
BOLD='\033[1m'
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

log_info "Starting API-Based MR-MT3 + Laplace Enhancement Pipeline"
log_info "Project root: $PROJECT_ROOT"
log_info "API endpoint: $API_URL"

# Create output directories
mkdir -p "$OUTPUT_DIR"/{api_transcriptions,enhanced_transcriptions,metrics,logs}
log_success "Created output directory structure"

# Convert paths to absolute
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
DATA_DIR="$(cd "$DATA_DIR" && pwd)"

# Check API health
log_info "Checking API availability..."
if ! HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/health"); then
    log_error "Failed to connect to API at $API_URL"
    exit 1
fi

if [ "$HTTP_CODE" != "200" ]; then
    log_error "API health check failed (HTTP $HTTP_CODE)"
    exit 1
fi

log_success "API is healthy and reachable"

################################################################################
# 2. Install Laplace Enhancement Dependencies Only
################################################################################

log_info "Installing Laplace enhancement dependencies..."
pip3 install --user \
    'numpy>=1.24.0,<2.0.0' \
    'scipy>=1.10.0,<1.15.0' \
    'librosa>=0.10.0,<0.11.0' \
    'pretty-midi>=0.2.10' \
    'mido>=1.3.0' \
    'matplotlib>=3.7.0,<4.0.0' \
    'PyYAML>=6.0' \
    'tqdm>=4.65.0' \
    2>&1 | tee "$OUTPUT_DIR/logs/dependencies_install.log"

log_success "Dependencies installed successfully"

################################################################################
# 3. Prepare Audio Files
################################################################################

log_info "Preparing audio files from $DATA_DIR..."

# Find all audio files
AUDIO_FILES=($(find "$DATA_DIR" -type f \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" -o -name "*.m4a" -o -name "*.ogg" \)))
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
# 4. Upload and Transcribe via API
################################################################################

log_info "Uploading and transcribing ${#AUDIO_FILES[@]} files via API..."

TRANSCRIPTION_DIR="$OUTPUT_DIR/api_transcriptions"
API_LOG="$OUTPUT_DIR/logs/api_operations.log"

TRANSCRIBED=0
FAILED=0

# Progress tracking
CURRENT_FILE=0
TOTAL_TO_PROCESS=${#AUDIO_FILES[@]}

echo -e "${BOLD}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║              API Transcription Progress                   ║${NC}"
echo -e "${BOLD}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

for AUDIO_FILE in "${AUDIO_FILES[@]}"; do
    CURRENT_FILE=$((CURRENT_FILE + 1))
    BASENAME=$(basename "$AUDIO_FILE")
    FILENAME_NO_EXT="${BASENAME%.*}"

    # Progress bar
    PERCENT=$((CURRENT_FILE * 100 / TOTAL_TO_PROCESS))
    FILLED=$((CURRENT_FILE * 40 / TOTAL_TO_PROCESS))
    EMPTY=$((40 - FILLED))
    BAR=$(printf "%${FILLED}s" | tr ' ' '█')
    SPACES=$(printf "%${EMPTY}s" | tr ' ' '░')

    echo -ne "\r${CYAN}[${BAR}${SPACES}] ${PERCENT}% (${CURRENT_FILE}/${TOTAL_TO_PROCESS})${NC} "
    echo -ne "${YELLOW}${BASENAME}${NC}                    \r"
    echo ""

    log_info "Processing: $BASENAME"

    # Step 1: Upload file
    log_info "  → Uploading to API..."
    UPLOAD_RESPONSE=$(curl -s -X POST "$API_URL/upload" \
        -F "file=@$AUDIO_FILE" \
        2>&1 | tee -a "$API_LOG")

    # Extract job_id from response
    JOB_ID=$(echo "$UPLOAD_RESPONSE" | grep -o '"job_id":"[^"]*"' | cut -d'"' -f4)

    if [ -z "$JOB_ID" ]; then
        log_error "  ✗ Upload failed for $BASENAME"
        ((FAILED++))
        continue
    fi

    log_success "  ✓ Uploaded (Job ID: $JOB_ID)"

    # Step 2: Start transcription
    log_info "  → Starting transcription..."
    PREDICT_RESPONSE=$(curl -s -X POST "$API_URL/predict/$JOB_ID" \
        -H "Content-Type: application/json" \
        -d "{
            \"confidence_threshold\": $CONFIDENCE,
            \"batch_size\": $BATCH_SIZE,
            \"use_stems\": false,
            \"output_format\": \"midi\"
        }" \
        2>&1 | tee -a "$API_LOG")

    log_success "  ✓ Transcription started"

    # Step 3: Poll for completion
    log_info "  → Waiting for completion..."
    MAX_WAIT=600  # 10 minutes
    ELAPSED=0

    while [ $ELAPSED -lt $MAX_WAIT ]; do
        STATUS_RESPONSE=$(curl -s "$API_URL/status/$JOB_ID" 2>&1 | tee -a "$API_LOG")
        STATUS=$(echo "$STATUS_RESPONSE" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)

        if [ "$STATUS" = "completed" ]; then
            log_success "  ✓ Transcription completed"
            break
        elif [ "$STATUS" = "failed" ]; then
            log_error "  ✗ Transcription failed"
            ((FAILED++))
            continue 2
        fi

        sleep 5
        ((ELAPSED+=5))
    done

    if [ $ELAPSED -ge $MAX_WAIT ]; then
        log_error "  ✗ Timeout waiting for transcription"
        ((FAILED++))
        continue
    fi

    # Step 4: Download MIDI file
    log_info "  → Downloading MIDI..."

    # Get results to find MIDI filename
    RESULTS_RESPONSE=$(curl -s "$API_URL/results/$JOB_ID" 2>&1 | tee -a "$API_LOG")
    MIDI_FILENAME=$(echo "$RESULTS_RESPONSE" | grep -o '"midi_full":"[^"]*"' | cut -d'"' -f4)

    if [ -z "$MIDI_FILENAME" ]; then
        log_error "  ✗ No MIDI file in results"
        ((FAILED++))
        continue
    fi

    # Download MIDI
    OUTPUT_MIDI="$TRANSCRIPTION_DIR/${FILENAME_NO_EXT}.mid"
    if curl -s "$API_URL/files/$MIDI_FILENAME" -o "$OUTPUT_MIDI"; then
        log_success "  ✓ Downloaded: $OUTPUT_MIDI"
        ((TRANSCRIBED++))
    else
        log_error "  ✗ Download failed"
        ((FAILED++))
    fi

    # Cleanup job (optional)
    curl -s -X DELETE "$API_URL/jobs/$JOB_ID" > /dev/null 2>&1
done

TRANSCRIPTION_COUNT=$(find "$TRANSCRIPTION_DIR" -name "*.mid" | wc -l)
log_success "API transcription complete: $TRANSCRIPTION_COUNT transcriptions ($FAILED failed)"

################################################################################
# 5. Run Laplace Enhancement
################################################################################

log_info "Running Laplace enhancement on transcriptions..."

cd "$PROJECT_ROOT"

ENHANCEMENT_DIR="$OUTPUT_DIR/enhanced_transcriptions"
ENHANCEMENT_LOG="$OUTPUT_DIR/logs/laplace_enhancement.log"

PROCESSED=0
FAILED=0

# Progress tracking
CURRENT_ENH=0
TOTAL_TO_ENHANCE=${#AUDIO_FILES[@]}

echo ""
echo -e "${BOLD}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║            Laplace Enhancement Progress                   ║${NC}"
echo -e "${BOLD}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

for AUDIO_FILE in "${AUDIO_FILES[@]}"; do
    CURRENT_ENH=$((CURRENT_ENH + 1))
    BASENAME=$(basename "$AUDIO_FILE" | sed 's/\.[^.]*$//')
    TRANSCRIPTION="$TRANSCRIPTION_DIR/${BASENAME}.mid"
    ENHANCED="$ENHANCEMENT_DIR/${BASENAME}_enhanced.mid"

    # Progress bar
    PERCENT_ENH=$((CURRENT_ENH * 100 / TOTAL_TO_ENHANCE))
    FILLED_ENH=$((CURRENT_ENH * 40 / TOTAL_TO_ENHANCE))
    EMPTY_ENH=$((40 - FILLED_ENH))
    BAR_ENH=$(printf "%${FILLED_ENH}s" | tr ' ' '█')
    SPACES_ENH=$(printf "%${EMPTY_ENH}s" | tr ' ' '░')

    echo -ne "\r${CYAN}[${BAR_ENH}${SPACES_ENH}] ${PERCENT_ENH}% (${CURRENT_ENH}/${TOTAL_TO_ENHANCE})${NC} "
    echo -ne "${YELLOW}${BASENAME}${NC}                    \r"
    echo ""

    if [ ! -f "$TRANSCRIPTION" ]; then
        log_warning "Transcription not found: $TRANSCRIPTION (skipping)"
        continue
    fi

    log_info "Enhancing: $BASENAME"

    # Run enhancement pipeline
    if python3 phase1_mrmt3_enhancement.py \
        --midi "$TRANSCRIPTION" \
        --audio "$AUDIO_FILE" \
        --output "$ENHANCED" \
        --config configs/enhancement.yaml \
        2>&1 | tee -a "$ENHANCEMENT_LOG"; then
        ((PROCESSED++))
        log_success "Enhanced: $BASENAME"
    else
        ((FAILED++))
        log_error "Failed: $BASENAME"
    fi
done

log_success "Laplace enhancement complete: $PROCESSED succeeded, $FAILED failed"

################################################################################
# 6. Generate Metrics Report
################################################################################

log_info "Generating evaluation metrics..."

METRICS_REPORT="$OUTPUT_DIR/metrics/evaluation_report.json"

python3 <<EOF
import json
from pathlib import Path

metrics = {
    "pipeline_version": "Phase1_API_MR-MT3_Laplace",
    "api_endpoint": "$API_URL",
    "total_files": ${#AUDIO_FILES[@]},
    "api_transcriptions": $TRANSCRIPTION_COUNT,
    "enhanced_transcriptions": $PROCESSED,
    "failed_enhancements": $FAILED,
    "processing_params": {
        "confidence_threshold": $CONFIDENCE,
        "batch_size": $BATCH_SIZE
    }
}

with open("$METRICS_REPORT", 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"Metrics saved to $METRICS_REPORT")
EOF

log_success "Metrics report generated"

################################################################################
# 7. Run Comparative Evaluation (Optional)
################################################################################

if [ -n "$RUN_EVALUATION" ]; then
    log_info "Running comparative evaluation..."

    EVAL_OUTPUT="$OUTPUT_DIR/evaluation"

    python3 evaluate_enhancement.py \
        --baseline-dir "$TRANSCRIPTION_DIR" \
        --enhanced-dir "$ENHANCEMENT_DIR" \
        --audio-dir "$DATA_DIR" \
        --output-dir "$EVAL_OUTPUT" \
        2>&1 | tee "$OUTPUT_DIR/logs/evaluation.log"

    if [ $? -eq 0 ]; then
        log_success "Evaluation complete!"
        log_info "View report: $EVAL_OUTPUT/evaluation_report.html"
    else
        log_warning "Evaluation failed (check logs)"
    fi
fi

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
echo "API transcripts:    $TRANSCRIPTION_COUNT"
echo "Enhanced:           $PROCESSED"
echo "Failed:             $FAILED"
echo ""
echo "Outputs:"
echo "  - API transcriptions:      $OUTPUT_DIR/api_transcriptions/"
echo "  - Enhanced transcriptions: $OUTPUT_DIR/enhanced_transcriptions/"
echo "  - Metrics:                 $OUTPUT_DIR/metrics/"
echo "  - Logs:                    $OUTPUT_DIR/logs/"
echo ""
echo "========================================================================"

log_success "Deployment complete!"
