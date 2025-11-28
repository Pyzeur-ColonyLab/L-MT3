#!/bin/bash
# =============================================================================
# Organize Processed Tracks - Create clean output folders
# =============================================================================
#
# Creates organized folders for each successfully processed track containing:
#   - Original audio (mix.wav)
#   - Enhanced MIDI
#   - Baseline MIDI (optional)
#   - Metadata (if available)
#
# Usage:
#   ./organize_processed_tracks.sh [--split validation] [--include-baseline]
#
# =============================================================================

set -e

# Configuration
DATASET_ROOT="${DATASET_ROOT:-$HOME/L-MT3/slakh2100_yourmt3_16k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/L-MT3/slakh2100_output}"
ORGANIZED_ROOT="${ORGANIZED_ROOT:-$HOME/L-MT3/organized_tracks}"

# Parse arguments
SPLIT="validation"
INCLUDE_BASELINE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --include-baseline)
            INCLUDE_BASELINE=true
            shift
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Usage: $0 [--split validation] [--include-baseline]"
            exit 1
            ;;
    esac
done

# Header
echo "======================================================================"
echo "ORGANIZE PROCESSED TRACKS"
echo "======================================================================"
echo ""
echo "📂 Source:     $DATASET_ROOT/$SPLIT"
echo "📁 Output:     $OUTPUT_ROOT/$SPLIT"
echo "📦 Organized:  $ORGANIZED_ROOT/$SPLIT"
echo "🎯 Split:      $SPLIT"
echo "📋 Baseline:   $([ "$INCLUDE_BASELINE" = true ] && echo "Included" || echo "Excluded")"
echo ""

# Check directories exist
if [ ! -d "$DATASET_ROOT/$SPLIT" ]; then
    echo "❌ Source directory not found: $DATASET_ROOT/$SPLIT"
    exit 1
fi

if [ ! -d "$OUTPUT_ROOT/$SPLIT" ]; then
    echo "❌ Output directory not found: $OUTPUT_ROOT/$SPLIT"
    exit 1
fi

# Create organized root directory
ORGANIZED_DIR="$ORGANIZED_ROOT/$SPLIT"
mkdir -p "$ORGANIZED_DIR"

echo "🔍 Scanning for processed tracks..."
echo ""

# Find all enhanced MIDI files (indicates successful processing)
ENHANCED_MIDIS=$(find "$OUTPUT_ROOT/$SPLIT" -name "*_enhanced.mid" 2>/dev/null)

if [ -z "$ENHANCED_MIDIS" ]; then
    echo "❌ No processed tracks found in $OUTPUT_ROOT/$SPLIT"
    exit 1
fi

# Count tracks
TOTAL_TRACKS=$(echo "$ENHANCED_MIDIS" | wc -l | tr -d ' ')
echo "✅ Found $TOTAL_TRACKS processed tracks"
echo ""

# Counters
SUCCESS=0
FAILED=0
CURRENT=0

echo "🚀 Organizing tracks..."
echo ""

# Process each enhanced MIDI
while IFS= read -r ENHANCED_MIDI; do
    CURRENT=$((CURRENT + 1))

    # Extract track name from path
    # Path format: .../Track00001/Track00001_enhanced.mid
    TRACK_DIR=$(dirname "$ENHANCED_MIDI")
    TRACK_NAME=$(basename "$TRACK_DIR")

    echo "[$CURRENT/$TOTAL_TRACKS] 📦 $TRACK_NAME"

    # Create organized track folder
    ORGANIZED_TRACK="$ORGANIZED_DIR/$TRACK_NAME"
    mkdir -p "$ORGANIZED_TRACK"

    # Source paths
    SOURCE_AUDIO="$DATASET_ROOT/$SPLIT/$TRACK_NAME/mix.wav"
    SOURCE_METADATA="$DATASET_ROOT/$SPLIT/$TRACK_NAME/metadata.yaml"
    BASELINE_MIDI="${ENHANCED_MIDI/_enhanced.mid/_baseline.mid}"

    # Copy enhanced MIDI
    if [ -f "$ENHANCED_MIDI" ]; then
        cp "$ENHANCED_MIDI" "$ORGANIZED_TRACK/${TRACK_NAME}_enhanced.mid"
        echo "  ✓ Enhanced MIDI copied"
    else
        echo "  ❌ Enhanced MIDI not found"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Copy original audio
    if [ -f "$SOURCE_AUDIO" ]; then
        cp "$SOURCE_AUDIO" "$ORGANIZED_TRACK/mix.wav"
        echo "  ✓ Original audio copied"
    else
        echo "  ⚠️  Original audio not found: $SOURCE_AUDIO"
    fi

    # Copy baseline MIDI if requested
    if [ "$INCLUDE_BASELINE" = true ] && [ -f "$BASELINE_MIDI" ]; then
        cp "$BASELINE_MIDI" "$ORGANIZED_TRACK/${TRACK_NAME}_baseline.mid"
        echo "  ✓ Baseline MIDI copied"
    fi

    # Copy metadata if available
    if [ -f "$SOURCE_METADATA" ]; then
        cp "$SOURCE_METADATA" "$ORGANIZED_TRACK/metadata.yaml"
        echo "  ✓ Metadata copied"
    fi

    # Create README for this track
    cat > "$ORGANIZED_TRACK/README.txt" << EOF
Track: $TRACK_NAME
Processed: $(date)
Split: $SPLIT

Files:
- mix.wav: Original audio input
- ${TRACK_NAME}_enhanced.mid: Laplace-enhanced transcription
$([ "$INCLUDE_BASELINE" = true ] && [ -f "$BASELINE_MIDI" ] && echo "- ${TRACK_NAME}_baseline.mid: MR-MT3 baseline transcription")
$([ -f "$SOURCE_METADATA" ] && echo "- metadata.yaml: Original track metadata")

Enhancement Pipeline:
1. MR-MT3 transcription (GPU inference)
2. Laplace feature extraction (decay rates, spectral analysis)
3. Instrument consolidation and timbre-based refinement
EOF

    SUCCESS=$((SUCCESS + 1))
    echo "  ✅ Complete"
    echo ""

done <<< "$ENHANCED_MIDIS"

# Final summary
echo "======================================================================"
echo "ORGANIZATION COMPLETE"
echo "======================================================================"
echo ""
echo "📊 Results:"
echo "  - Successfully organized: $SUCCESS tracks"
echo "  - Failed: $FAILED tracks"
echo ""
echo "📁 Output location:"
echo "  $ORGANIZED_DIR"
echo ""
echo "📂 Folder structure:"
echo "  $ORGANIZED_DIR/"
echo "  ├── Track01647/"
echo "  │   ├── mix.wav"
echo "  │   ├── Track01647_enhanced.mid"
if [ "$INCLUDE_BASELINE" = true ]; then
echo "  │   ├── Track01647_baseline.mid"
fi
echo "  │   ├── metadata.yaml (if available)"
echo "  │   └── README.txt"
echo "  ├── Track01648/"
echo "  └── ..."
echo ""

# Create master index
INDEX_FILE="$ORGANIZED_DIR/INDEX.txt"
cat > "$INDEX_FILE" << EOF
Organized Processed Tracks - Index
==================================
Split: $SPLIT
Generated: $(date)
Total tracks: $SUCCESS

Track List:
-----------
EOF

# List all organized tracks
ls -1 "$ORGANIZED_DIR" | grep "^Track" | sort >> "$INDEX_FILE"

echo "📋 Index created: $INDEX_FILE"
echo ""

# Calculate total size
TOTAL_SIZE=$(du -sh "$ORGANIZED_DIR" | cut -f1)
echo "💾 Total size: $TOTAL_SIZE"
echo ""

echo "======================================================================"

# Exit code
if [ $SUCCESS -eq 0 ]; then
    exit 1
else
    exit 0
fi
