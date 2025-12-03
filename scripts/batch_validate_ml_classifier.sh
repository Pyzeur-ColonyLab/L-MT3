#!/bin/bash
# =============================================================================
# Batch Validation: ML Classifier Performance on Slakh2100 Validation Set
# =============================================================================
#
# Processes all tracks in slakh2100_yourmt3_16k/validation with Phase 2 ML
# classifier and collects comprehensive metrics for analysis.
#
# Usage:
#   ./batch_validate_ml_classifier.sh [OPTIONS]
#
# Options:
#   --max-tracks N     Process only first N tracks (default: all)
#   --confidence CONF  Confidence threshold (default: 0.7)
#   --output-dir DIR   Output directory (default: ./validation_results)
#   --skip-existing    Skip tracks that already have output files
#
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Paths
AUDIO_BASE="${PROJECT_DIR}/slakh2100_yourmt3_16k/validation"
MIDI_BASE="${PROJECT_DIR}/slakh2100_output/validation"
CLASSIFIER_PATH="${PROJECT_DIR}/laplace_classifier.pkl"
ENHANCEMENT_SCRIPT="${PROJECT_DIR}/phase1_mrmt3_enhancement.py"

# Default parameters
MAX_TRACKS=""
CONFIDENCE=0.7
OUTPUT_DIR="${PROJECT_DIR}/validation_results"
SKIP_EXISTING=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-tracks)
            MAX_TRACKS="$2"
            shift 2
            ;;
        --confidence)
            CONFIDENCE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/enhanced_midi"
mkdir -p "$OUTPUT_DIR/logs"

RESULTS_CSV="${OUTPUT_DIR}/validation_results.csv"
SUMMARY_FILE="${OUTPUT_DIR}/summary.txt"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${OUTPUT_DIR}/logs/batch_run_${TIMESTAMP}.log"

# =============================================================================
# Utility Functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" | tee -a "$LOG_FILE" >&2
}

# =============================================================================
# Initialize Results CSV
# =============================================================================

if [ ! -f "$RESULTS_CSV" ]; then
    log "Creating results CSV: $RESULTS_CSV"
    cat > "$RESULTS_CSV" << 'EOF'
track_id,status,total_time_s,feature_extraction_s,consolidation_s,refinement_s,input_instruments,output_instruments,instrument_reduction_pct,refinements_applied,input_notes,output_notes,duration_s,error_message
EOF
fi

# =============================================================================
# Find All Tracks
# =============================================================================

log "========================================================================"
log "BATCH VALIDATION: ML CLASSIFIER ON SLAKH2100"
log "========================================================================"
log ""
log "Configuration:"
log "  Audio base: $AUDIO_BASE"
log "  MIDI base: $MIDI_BASE"
log "  Classifier: $CLASSIFIER_PATH"
log "  Output directory: $OUTPUT_DIR"
log "  Confidence threshold: $CONFIDENCE"
log "  Max tracks: ${MAX_TRACKS:-all}"
log "  Skip existing: $SKIP_EXISTING"
log ""

# Find all track directories
TRACK_DIRS=($(find "$AUDIO_BASE" -maxdepth 1 -type d -name "Track*" | sort))
TOTAL_TRACKS=${#TRACK_DIRS[@]}

if [ "$MAX_TRACKS" != "" ] && [ "$MAX_TRACKS" -lt "$TOTAL_TRACKS" ]; then
    TRACK_DIRS=("${TRACK_DIRS[@]:0:$MAX_TRACKS}")
    TOTAL_TRACKS=$MAX_TRACKS
fi

log "Found $TOTAL_TRACKS tracks to process"
log ""

# =============================================================================
# Process Each Track
# =============================================================================

PROCESSED=0
SUCCESS=0
FAILED=0
SKIPPED=0

START_TIME=$(date +%s)

for TRACK_DIR in "${TRACK_DIRS[@]}"; do
    TRACK_ID=$(basename "$TRACK_DIR")

    log "========================================================================"
    log "Processing $TRACK_ID ($((PROCESSED + 1))/$TOTAL_TRACKS)"
    log "========================================================================"

    # Check paths
    AUDIO_PATH="$TRACK_DIR/mix.wav"
    MIDI_PATH="$MIDI_BASE/$TRACK_ID/${TRACK_ID}_baseline.mid"
    OUTPUT_PATH="$OUTPUT_DIR/enhanced_midi/${TRACK_ID}_enhanced.mid"
    TRACK_LOG="$OUTPUT_DIR/logs/${TRACK_ID}_${TIMESTAMP}.log"

    # Validate input files exist
    if [ ! -f "$AUDIO_PATH" ]; then
        log_error "Audio file not found: $AUDIO_PATH"
        echo "$TRACK_ID,failed,0,0,0,0,0,0,0,0,0,0,0,Audio file not found" >> "$RESULTS_CSV"
        FAILED=$((FAILED + 1))
        PROCESSED=$((PROCESSED + 1))
        continue
    fi

    if [ ! -f "$MIDI_PATH" ]; then
        log_error "MIDI file not found: $MIDI_PATH"
        echo "$TRACK_ID,failed,0,0,0,0,0,0,0,0,0,0,0,MIDI file not found" >> "$RESULTS_CSV"
        FAILED=$((FAILED + 1))
        PROCESSED=$((PROCESSED + 1))
        continue
    fi

    # Skip if output exists and --skip-existing flag set
    if [ "$SKIP_EXISTING" = true ] && [ -f "$OUTPUT_PATH" ]; then
        log "Skipping $TRACK_ID (output exists)"
        SKIPPED=$((SKIPPED + 1))
        PROCESSED=$((PROCESSED + 1))
        continue
    fi

    # Run enhancement
    TRACK_START=$(date +%s)

    log "Running enhancement..."
    if python3 "$ENHANCEMENT_SCRIPT" \
        --midi "$MIDI_PATH" \
        --audio "$AUDIO_PATH" \
        --output "$OUTPUT_PATH" \
        --use-ml-classifier \
        --classifier-path "$CLASSIFIER_PATH" \
        --verbose \
        > "$TRACK_LOG" 2>&1; then

        TRACK_END=$(date +%s)
        TRACK_TIME=$((TRACK_END - TRACK_START))

        log "✓ Enhancement completed in ${TRACK_TIME}s"

        # Parse metrics from log file
        TOTAL_TIME=$(grep "Enhancement Complete in" "$TRACK_LOG" | grep -oP '\d+\.\d+' || echo "0")
        FEATURE_TIME=$(grep "Feature extraction:" "$TRACK_LOG" | grep -oP '\d+\.\d+' || echo "0")
        CONSOLIDATION_TIME=$(grep "Consolidation:" "$TRACK_LOG" | grep -oP '\d+\.\d+' || echo "0")
        REFINEMENT_TIME=$(grep "Refinement:" "$TRACK_LOG" | grep -oP '\d+\.\d+' || echo "0")

        INPUT_INSTRUMENTS=$(grep "Input: .* instruments" "$TRACK_LOG" | grep -oP 'Instruments: \K\d+' || echo "0")
        OUTPUT_INSTRUMENTS=$(grep "Output: .* instruments" "$TRACK_LOG" | grep -oP 'Output: \K\d+' || echo "0")

        REFINEMENTS=$(grep "Refinement complete:" "$TRACK_LOG" | grep -oP '\d+(?=/\d+ instruments modified)' || echo "0")

        INPUT_NOTES=$(grep "Notes: \d+" "$TRACK_LOG" | head -1 | grep -oP '\d+' || echo "0")
        OUTPUT_NOTES=$(grep "Notes: \d+ \(reduced by" "$TRACK_LOG" | grep -oP 'Notes: \K\d+' || echo "$INPUT_NOTES")

        DURATION=$(grep "Duration: \d+\.\d+s" "$TRACK_LOG" | head -1 | grep -oP '\d+\.\d+' || echo "0")

        # Calculate reduction percentage
        if [ "$INPUT_INSTRUMENTS" -gt 0 ]; then
            REDUCTION=$(echo "scale=2; (($INPUT_INSTRUMENTS - $OUTPUT_INSTRUMENTS) / $INPUT_INSTRUMENTS) * 100" | bc)
        else
            REDUCTION="0"
        fi

        # Write to CSV
        echo "$TRACK_ID,success,$TOTAL_TIME,$FEATURE_TIME,$CONSOLIDATION_TIME,$REFINEMENT_TIME,$INPUT_INSTRUMENTS,$OUTPUT_INSTRUMENTS,$REDUCTION,$REFINEMENTS,$INPUT_NOTES,$OUTPUT_NOTES,$DURATION," >> "$RESULTS_CSV"

        SUCCESS=$((SUCCESS + 1))

        log "Metrics:"
        log "  Input instruments: $INPUT_INSTRUMENTS"
        log "  Output instruments: $OUTPUT_INSTRUMENTS"
        log "  Reduction: ${REDUCTION}%"
        log "  ML refinements: $REFINEMENTS"
        log "  Processing time: ${TOTAL_TIME}s"

    else
        TRACK_END=$(date +%s)
        TRACK_TIME=$((TRACK_END - TRACK_START))

        # Extract error message
        ERROR_MSG=$(tail -5 "$TRACK_LOG" | grep -oP '(ERROR|Error|error): \K.*' | head -1 | tr ',' ';' || echo "Unknown error")

        log_error "Enhancement failed after ${TRACK_TIME}s"
        log_error "Error: $ERROR_MSG"

        echo "$TRACK_ID,failed,0,0,0,0,0,0,0,0,0,0,0,$ERROR_MSG" >> "$RESULTS_CSV"

        FAILED=$((FAILED + 1))
    fi

    PROCESSED=$((PROCESSED + 1))
    log ""
done

# =============================================================================
# Generate Summary
# =============================================================================

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

log "========================================================================"
log "BATCH PROCESSING COMPLETE"
log "========================================================================"
log ""
log "Summary:"
log "  Total tracks processed: $PROCESSED"
log "  Successful: $SUCCESS"
log "  Failed: $FAILED"
log "  Skipped: $SKIPPED"
log "  Total duration: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
log ""
log "Results saved to: $RESULTS_CSV"
log "Logs saved to: $OUTPUT_DIR/logs/"
log ""

# Generate detailed summary file
cat > "$SUMMARY_FILE" << EOF
======================================================================
BATCH VALIDATION SUMMARY
======================================================================

Run timestamp: $(date '+%Y-%m-%d %H:%M:%S')

Configuration:
  Confidence threshold: $CONFIDENCE
  Max tracks: ${MAX_TRACKS:-all}
  Total processed: $PROCESSED

Results:
  ✓ Successful: $SUCCESS
  ✗ Failed: $FAILED
  ⊘ Skipped: $SKIPPED

Processing time:
  Total: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s
  Average per track: $(echo "scale=2; $TOTAL_DURATION / $PROCESSED" | bc)s

Output files:
  Results CSV: $RESULTS_CSV
  Enhanced MIDIs: $OUTPUT_DIR/enhanced_midi/
  Logs: $OUTPUT_DIR/logs/

======================================================================
DETAILED STATISTICS (from successful runs)
======================================================================

EOF

# Calculate statistics using Python
python3 - "$RESULTS_CSV" >> "$SUMMARY_FILE" << 'PYTHON_SCRIPT'
import sys
import csv
from statistics import mean, median, stdev

csv_file = sys.argv[1]

with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    rows = [row for row in reader if row['status'] == 'success']

if not rows:
    print("No successful runs to analyze")
    sys.exit(0)

# Extract numeric data
times = [float(row['total_time_s']) for row in rows if row['total_time_s']]
feature_times = [float(row['feature_extraction_s']) for row in rows if row['feature_extraction_s']]
consolidation_times = [float(row['consolidation_s']) for row in rows if row['consolidation_s']]
refinement_times = [float(row['refinement_s']) for row in rows if row['refinement_s']]

input_insts = [int(row['input_instruments']) for row in rows if row['input_instruments']]
output_insts = [int(row['output_instruments']) for row in rows if row['output_instruments']]
reductions = [float(row['instrument_reduction_pct']) for row in rows if row['instrument_reduction_pct']]
refinements = [int(row['refinements_applied']) for row in rows if row['refinements_applied']]

print(f"Processing Times (seconds):")
print(f"  Total time:           mean={mean(times):.1f}  median={median(times):.1f}  stdev={stdev(times) if len(times) > 1 else 0:.1f}")
print(f"  Feature extraction:   mean={mean(feature_times):.1f}  median={median(feature_times):.1f}")
print(f"  Consolidation:        mean={mean(consolidation_times):.1f}  median={median(consolidation_times):.1f}")
print(f"  Refinement:           mean={mean(refinement_times):.1f}  median={median(refinement_times):.1f}")
print()

print(f"Instrument Statistics:")
print(f"  Input instruments:    mean={mean(input_insts):.1f}  median={median(input_insts):.0f}  range={min(input_insts)}-{max(input_insts)}")
print(f"  Output instruments:   mean={mean(output_insts):.1f}  median={median(output_insts):.0f}  range={min(output_insts)}-{max(output_insts)}")
print(f"  Reduction:            mean={mean(reductions):.1f}%  median={median(reductions):.1f}%")
print()

print(f"ML Refinement Statistics:")
print(f"  Refinements applied:  mean={mean(refinements):.1f}  median={median(refinements):.0f}  range={min(refinements)}-{max(refinements)}")
print(f"  Tracks with refinements: {sum(1 for r in refinements if r > 0)} / {len(refinements)} ({sum(1 for r in refinements if r > 0) / len(refinements) * 100:.1f}%)")
print()

print(f"Distribution of refinements per track:")
from collections import Counter
dist = Counter(refinements)
for count in sorted(dist.keys()):
    print(f"  {count} refinements: {dist[count]} tracks ({dist[count] / len(refinements) * 100:.1f}%)")
PYTHON_SCRIPT

log "Summary saved to: $SUMMARY_FILE"
log ""
log "To view results:"
log "  cat $SUMMARY_FILE"
log "  cat $RESULTS_CSV"
log ""
log "========================================================================"
