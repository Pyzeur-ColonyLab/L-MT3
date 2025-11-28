#!/bin/bash
# =============================================================================
# MR-MT3 + Laplace Enhancement Pipeline - Batch Processing with Metrics
# =============================================================================
#
# Processes all tracks in BabySLAKH dataset with metrics recording
#
# Features:
# - Resume capability (skip already processed tracks)
# - Progress tracking with time estimates
# - Error handling and logging
# - Metrics recording for analysis
# - Summary report generation
#
# Usage:
#   ./run_batch_pipeline_with_metrics.sh [OPTIONS]
#
# Options:
#   --dataset DIR     Dataset directory (default: ~/L-MT3/babyslakh_16k/)
#   --output DIR      Output directory (default: ~/L-MT3/batch_output/)
#   --metrics DIR     Metrics directory (default: ~/L-MT3/metrics/)
#   --resume          Resume from previous run (default: yes)
#   --clean           Clean previous run and start fresh
#
# =============================================================================

set -e  # Exit on error

# Configuration
DATASET_DIR="${1:-$HOME/L-MT3/babyslakh_16k/}"
OUTPUT_DIR="${2:-$HOME/L-MT3/batch_output/}"
METRICS_DIR="${3:-$HOME/L-MT3/metrics/}"
RESUME_FILE="$OUTPUT_DIR/.resume_state"
LOG_DIR="$OUTPUT_DIR/logs"

# Parse arguments
CLEAN_START=false
RESUME_MODE=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --metrics)
            METRICS_DIR="$2"
            shift 2
            ;;
        --clean)
            CLEAN_START=true
            shift
            ;;
        --no-resume)
            RESUME_MODE=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$METRICS_DIR"

# Clean previous run if requested
if [ "$CLEAN_START" = true ]; then
    echo "🧹 Cleaning previous run..."
    rm -f "$RESUME_FILE"
    rm -rf "$METRICS_DIR"/*
    mkdir -p "$METRICS_DIR"
fi

# Header
echo "=" | awk '{for(i=1;i<=70;i++) printf "="; printf "\n"}'
echo "MR-MT3 + LAPLACE BATCH PROCESSING WITH METRICS"
echo "=" | awk '{for(i=1;i<=70;i++) printf "="; printf "\n"}'
echo ""
echo "📂 Dataset: $DATASET_DIR"
echo "📁 Output:  $OUTPUT_DIR"
echo "📊 Metrics: $METRICS_DIR"
echo ""

# Check dataset exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "❌ Dataset directory not found: $DATASET_DIR"
    exit 1
fi

# Count total tracks
TOTAL_TRACKS=$(find "$DATASET_DIR" -type d -name "Track*" | wc -l)
echo "📊 Found $TOTAL_TRACKS tracks to process"

# Load resume state
PROCESSED=""
if [ "$RESUME_MODE" = true ] && [ -f "$RESUME_FILE" ]; then
    PROCESSED=$(cat "$RESUME_FILE")
    SKIPPED=$(echo "$PROCESSED" | wc -w)
    echo "📂 Resuming from previous run (${SKIPPED} tracks already processed)"
fi

# Counters
SUCCESS=0
FAILED=0
SKIPPED=0
CURRENT=0

# Processing start time
BATCH_START=$(date +%s)

echo ""
echo "🚀 Starting batch processing..."
echo ""

# Process each track
for TRACK_DIR in "$DATASET_DIR"/Track*; do
    TRACK_NAME=$(basename "$TRACK_DIR")
    CURRENT=$((CURRENT + 1))

    # Check if already processed (resume)
    if echo "$PROCESSED" | grep -q "$TRACK_NAME"; then
        SKIPPED=$((SKIPPED + 1))
        echo "[$CURRENT/$TOTAL_TRACKS] ⏩ Skipping $TRACK_NAME (already processed)"
        continue
    fi

    # Input files
    AUDIO_FILE="$TRACK_DIR/mix.wav"
    if [ ! -f "$AUDIO_FILE" ]; then
        echo "[$CURRENT/$TOTAL_TRACKS] ⚠️  $TRACK_NAME: mix.wav not found, skipping"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Output directory for this track
    TRACK_OUTPUT="$OUTPUT_DIR/$TRACK_NAME"
    mkdir -p "$TRACK_OUTPUT"

    # MIDI output path
    BASELINE_MIDI="$TRACK_OUTPUT/${TRACK_NAME}_baseline.mid"
    ENHANCED_MIDI="$TRACK_OUTPUT/${TRACK_NAME}_enhanced.mid"

    # Log file
    LOG_FILE="$LOG_DIR/${TRACK_NAME}.log"

    echo "[$CURRENT/$TOTAL_TRACKS] 🎵 Processing $TRACK_NAME..."

    # Track start time
    TRACK_START=$(date +%s)

    # Step 1: MR-MT3 Transcription
    echo "  [1/2] Running MR-MT3 transcription..."
    if python3 ~/L-MT3/run_mrmt3_inference.py \
        --model "$HOME/models/mr-mt3/mt3.pth" \
        --audio "$AUDIO_FILE" \
        --output "$BASELINE_MIDI" \
        --device cuda > "$LOG_FILE" 2>&1; then

        # Step 2: Laplace Enhancement with metrics
        echo "  [2/2] Running Laplace enhancement with metrics recording..."
        cd ~/L-MT3
        if python3 research/phase1_mrmt3_enhancement.py \
            --midi "$BASELINE_MIDI" \
            --audio "$AUDIO_FILE" \
            --output "$ENHANCED_MIDI" \
            --metrics-dir "$METRICS_DIR" >> "$LOG_FILE" 2>&1; then

            # Success
            SUCCESS=$((SUCCESS + 1))

            # Track processing time
            TRACK_END=$(date +%s)
            TRACK_DURATION=$((TRACK_END - TRACK_START))

            # Update resume file
            echo "$TRACK_NAME" >> "$RESUME_FILE"

            echo "  ✅ Complete in ${TRACK_DURATION}s"
        else
            FAILED=$((FAILED + 1))
            echo "  ❌ Enhancement failed (see log: $LOG_FILE)"
        fi
    else
        FAILED=$((FAILED + 1))
        echo "  ❌ MR-MT3 transcription failed (see log: $LOG_FILE)"
    fi

    # Progress summary
    COMPLETED=$((SUCCESS + FAILED))
    PROGRESS=$(awk "BEGIN {printf \"%.1f\", ($COMPLETED / $TOTAL_TRACKS) * 100}")

    # Time estimate
    if [ $COMPLETED -gt 0 ]; then
        BATCH_ELAPSED=$(($(date +%s) - BATCH_START))
        AVG_TIME=$((BATCH_ELAPSED / COMPLETED))
        REMAINING=$((TOTAL_TRACKS - COMPLETED - SKIPPED))
        ETA=$((REMAINING * AVG_TIME))
        ETA_MIN=$((ETA / 60))

        echo "  📊 Progress: $COMPLETED/$TOTAL_TRACKS ($PROGRESS%) | Success: $SUCCESS | Failed: $FAILED | ETA: ${ETA_MIN}min"
    fi

    echo ""
done

# Final summary
echo "=" | awk '{for(i=1;i<=70;i++) printf "="; printf "\n"}'
echo "BATCH PROCESSING COMPLETE"
echo "=" | awk '{for(i=1;i<=70;i++) printf "="; printf "\n"}'
echo ""
echo "📊 Results:"
echo "  - Total tracks:    $TOTAL_TRACKS"
echo "  - Successful:      $SUCCESS"
echo "  - Failed:          $FAILED"
echo "  - Skipped:         $SKIPPED"
echo ""

# Total time
BATCH_END=$(date +%s)
BATCH_DURATION=$((BATCH_END - BATCH_START))
BATCH_HOURS=$((BATCH_DURATION / 3600))
BATCH_MINS=$(((BATCH_DURATION % 3600) / 60))
echo "⏱️  Total time: ${BATCH_HOURS}h ${BATCH_MINS}m"
echo ""

# Generate and display metrics summary
if [ $SUCCESS -gt 0 ]; then
    echo "📈 Generating metrics summary..."
    echo ""
    cd ~/L-MT3
    python3 research/show_metrics_summary.py --metrics-dir "$METRICS_DIR"
else
    echo "⚠️  No successful tracks to analyze"
fi

echo ""
echo "📁 Output files:"
echo "  - MIDI files:  $OUTPUT_DIR"
echo "  - Logs:        $LOG_DIR"
echo "  - Metrics CSV: $METRICS_DIR/pipeline_metrics.csv"
echo "  - Metrics JSON: $METRICS_DIR/pipeline_metrics.jsonl"
echo "  - Summary:     $METRICS_DIR/batch_summary.json"
echo ""
echo "=" | awk '{for(i=1;i<=70;i++) printf "="; printf "\n"}'

# Exit code based on results
if [ $SUCCESS -eq 0 ]; then
    exit 1  # All failed
elif [ $FAILED -gt 0 ]; then
    exit 2  # Some failed
else
    exit 0  # All succeeded
fi
