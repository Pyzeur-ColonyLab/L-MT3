#!/bin/bash
#
# Real Music Testing Helper Script for L-MT3 Laplace Classifier
#
# This script simplifies testing the trained classifier on real music files.
# It handles the complete pipeline: MR-MT3 inference → ML enhancement → per-instrument export
#
# Usage:
#   ./run_real_music_test.sh <audio_dir> [output_dir] [classifier_path] [model_path]
#
# Examples:
#   # Process directory with defaults
#   ./run_real_music_test.sh ./music_samples
#
#   # Custom output and classifier paths
#   ./run_real_music_test.sh ./music_samples ./my_results ./my_classifier.pkl
#
#   # Full custom paths
#   ./run_real_music_test.sh ./music_samples ./results ./laplace_classifier.pkl ./models/mr_mt3.pth

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script banner
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  L-MT3 Real Music Testing - Laplace Classifier${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Parse arguments
AUDIO_DIR="${1:-./music_samples}"
OUTPUT_DIR="${2:-./test_results}"
CLASSIFIER_PATH="${3:-./laplace_classifier.pkl}"
MODEL_PATH="${4:-}"

# Validate inputs
if [ ! -d "$AUDIO_DIR" ]; then
    echo -e "${RED}✗ Error: Audio directory not found: $AUDIO_DIR${NC}"
    echo ""
    echo "Usage: $0 <audio_dir> [output_dir] [classifier_path] [model_path]"
    exit 1
fi

if [ ! -f "$CLASSIFIER_PATH" ]; then
    echo -e "${RED}✗ Error: Classifier not found: $CLASSIFIER_PATH${NC}"
    echo ""
    echo "Please train the classifier first:"
    echo "  sudo ./scripts/setup_phase2_training.sh"
    exit 1
fi

# Count audio files
AUDIO_COUNT=$(find "$AUDIO_DIR" -type f \( -iname "*.wav" -o -iname "*.mp3" -o -iname "*.flac" -o -iname "*.ogg" -o -iname "*.m4a" \) | wc -l | tr -d ' ')

if [ "$AUDIO_COUNT" -eq 0 ]; then
    echo -e "${RED}✗ Error: No audio files found in $AUDIO_DIR${NC}"
    echo ""
    echo "Supported formats: .wav, .mp3, .flac, .ogg, .m4a"
    exit 1
fi

# Display configuration
echo -e "${GREEN}Configuration:${NC}"
echo "  Audio directory:  $AUDIO_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Classifier:       $CLASSIFIER_PATH"
if [ -n "$MODEL_PATH" ]; then
    echo "  MR-MT3 model:     $MODEL_PATH"
else
    echo "  MR-MT3 model:     (expecting existing baseline MIDI files)"
fi
echo ""
echo -e "${GREEN}Found:${NC} $AUDIO_COUNT audio files"
echo ""

# Confirm
read -p "$(echo -e ${YELLOW}Continue with processing? [y/N]:${NC} )" -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Cancelled${NC}"
    exit 0
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Starting Processing${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Build command
CMD="python3 test_real_music.py --audio-dir \"$AUDIO_DIR\" --output-dir \"$OUTPUT_DIR\" --classifier \"$CLASSIFIER_PATH\""

if [ -n "$MODEL_PATH" ]; then
    CMD="$CMD --model \"$MODEL_PATH\""
fi

CMD="$CMD --verbose"

# Run processing
echo -e "${GREEN}Running:${NC} $CMD"
echo ""

eval $CMD
EXIT_CODE=$?

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════${NC}"

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Processing Complete${NC}"
    echo ""
    echo -e "${GREEN}Results:${NC}"
    echo "  Directory: $OUTPUT_DIR"
    echo "  Summary:   $OUTPUT_DIR/summary_report.txt"
    echo ""
    echo "Per-track results:"
    for dir in "$OUTPUT_DIR"/*/ ; do
        if [ -d "$dir" ]; then
            track_name=$(basename "$dir")
            echo "  • $track_name/"
            echo "      Enhanced MIDI:    ${track_name}_enhanced.mid"
            echo "      Instruments dir:  instruments/"
            echo "      Report:           ${track_name}_report.txt"
        fi
    done
    echo ""
    echo -e "${GREEN}View summary report:${NC}"
    echo "  cat $OUTPUT_DIR/summary_report.txt"
else
    echo -e "${RED}✗ Processing Failed${NC}"
    echo ""
    echo "Check logs above for errors."
    echo "For more details, run with --verbose flag"
fi

echo -e "${BLUE}════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

exit $EXIT_CODE
