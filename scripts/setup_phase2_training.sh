#!/bin/bash
# =============================================================================
# Phase 2 Setup: Laplace Classifier Training on GPU Instance
# =============================================================================
#
# Prepares GPU instance for NSynth dataset processing and classifier training.
# Downloads dataset, extracts features, and trains ML model.
#
# Usage:
#   sudo ./setup_phase2_training.sh
#
# Expected runtime: 3-4 hours
# Storage required: 35GB
# =============================================================================

set -e

echo "======================================================================"
echo "PHASE 2 SETUP: LAPLACE CLASSIFIER TRAINING"
echo "======================================================================"
echo ""

# Configuration
WORK_DIR="${HOME}/L-MT3"
NSYNTH_DIR="${WORK_DIR}/nsynth-train"
FEATURES_FILE="${WORK_DIR}/nsynth_features.npz"
MODEL_OUTPUT="${WORK_DIR}/laplace_classifier.pkl"
LOG_DIR="${WORK_DIR}/phase2_logs"

# Create directories
mkdir -p "$LOG_DIR"
cd "$WORK_DIR"

echo "📂 Working directory: $WORK_DIR"
echo "💾 Storage check..."
AVAILABLE_GB=$(df -BG "$WORK_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
echo "   Available: ${AVAILABLE_GB}GB (35GB required)"

if [ "$AVAILABLE_GB" -lt 35 ]; then
    echo "❌ Insufficient storage. Need 35GB, have ${AVAILABLE_GB}GB"
    exit 1
fi

# =============================================================================
# STEP 1: Install Dependencies
# =============================================================================

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "STEP 1: Installing Dependencies"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

# Activate venv
if [ -d "$WORK_DIR/venv" ]; then
    source "$WORK_DIR/venv/bin/activate"
    echo "✓ Activated existing venv"
else
    echo "Creating new virtual environment..."
    python3 -m venv "$WORK_DIR/venv"
    source "$WORK_DIR/venv/bin/activate"
fi

# Install classifier dependencies
echo "Installing Python packages..."
pip install -q --upgrade pip
pip install -q scikit-learn xgboost matplotlib seaborn joblib

echo "✓ Dependencies installed"

# =============================================================================
# STEP 2: Download NSynth Dataset
# =============================================================================

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "STEP 2: Downloading NSynth Dataset"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

if [ -d "$NSYNTH_DIR" ] && [ "$(ls -A $NSYNTH_DIR)" ]; then
    echo "✓ NSynth dataset already exists at $NSYNTH_DIR"
    TRACK_COUNT=$(find "$NSYNTH_DIR" -name "*.wav" | wc -l)
    echo "  Found $TRACK_COUNT audio files"
else
    echo "📥 Downloading NSynth train set (~25GB)..."
    echo "   This will take 15-30 minutes depending on connection speed"

    NSYNTH_URL="https://storage.googleapis.com/magentadata/datasets/nsynth/nsynth-train.jsonwav.tar.gz"
    NSYNTH_ARCHIVE="${WORK_DIR}/nsynth-train.jsonwav.tar.gz"

    # Download with progress
    wget --progress=bar:force:noscroll -O "$NSYNTH_ARCHIVE" "$NSYNTH_URL" 2>&1 | \
        grep -oP '\d+%' | \
        while read line; do echo "   Progress: $line"; done

    echo ""
    echo "📦 Extracting archive..."
    tar -xzf "$NSYNTH_ARCHIVE" -C "$WORK_DIR"

    echo "🗑️  Cleaning up archive..."
    rm "$NSYNTH_ARCHIVE"

    TRACK_COUNT=$(find "$NSYNTH_DIR" -name "*.wav" | wc -l)
    echo "✓ Extracted $TRACK_COUNT audio files"
fi

# =============================================================================
# STEP 3: Extract Laplace Features
# =============================================================================

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "STEP 3: Extracting Laplace Features"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

if [ -f "$FEATURES_FILE" ]; then
    echo "✓ Features file already exists: $FEATURES_FILE"
    FILE_SIZE=$(du -h "$FEATURES_FILE" | cut -f1)
    echo "  Size: $FILE_SIZE"
else
    echo "🔬 Extracting 26 Laplace features from NSynth samples..."
    echo "   This will take 2-3 hours"
    echo "   Using all available CPU cores"

    START_TIME=$(date +%s)

    # Run feature extraction with all cores
    N_CORES=$(nproc)
    python3 "$WORK_DIR/Laplace_classifier/extract_nsynth_features.py" \
        --nsynth_dir "$NSYNTH_DIR" \
        --output "$FEATURES_FILE" \
        --n_workers "$N_CORES" \
        > "$LOG_DIR/feature_extraction.log" 2>&1

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    DURATION_MIN=$((DURATION / 60))

    echo "✓ Feature extraction complete in ${DURATION_MIN} minutes"
    FILE_SIZE=$(du -h "$FEATURES_FILE" | cut -f1)
    echo "  Features saved: $FEATURES_FILE ($FILE_SIZE)"
fi

# =============================================================================
# STEP 4: Train Classifiers
# =============================================================================

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "STEP 4: Training Classifiers"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

echo "📊 Training multiple models for comparison..."

# Train Random Forest
echo ""
echo "--- Random Forest ---"
RF_MODEL="${WORK_DIR}/laplace_classifier_rf.pkl"
python3 "$WORK_DIR/Laplace_classifier/train_classifier.py" \
    --features "$FEATURES_FILE" \
    --output "$RF_MODEL" \
    --model_type random_forest \
    --no_plot \
    > "$LOG_DIR/train_rf.log" 2>&1

RF_ACC=$(grep "Val accuracy:" "$LOG_DIR/train_rf.log" | tail -1 | grep -oP '\d+\.\d+%')
echo "✓ Random Forest trained: $RF_ACC accuracy"

# Train XGBoost
echo ""
echo "--- XGBoost ---"
XGB_MODEL="${WORK_DIR}/laplace_classifier_xgb.pkl"
python3 "$WORK_DIR/Laplace_classifier/train_classifier.py" \
    --features "$FEATURES_FILE" \
    --output "$XGB_MODEL" \
    --model_type xgboost \
    --no_plot \
    > "$LOG_DIR/train_xgb.log" 2>&1

XGB_ACC=$(grep "Val accuracy:" "$LOG_DIR/train_xgb.log" | tail -1 | grep -oP '\d+\.\d+%')
echo "✓ XGBoost trained: $XGB_ACC accuracy"

# Select best model as default
echo ""
echo "🏆 Selecting best model as default..."
cp "$XGB_MODEL" "$MODEL_OUTPUT"
echo "✓ Best model saved as: $MODEL_OUTPUT"

# =============================================================================
# STEP 5: Validation
# =============================================================================

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "STEP 5: Model Validation"
echo "══════════════════════════════════════════════════════════════════════"
echo ""

echo "🧪 Running quick validation test..."
python3 -c "
from Laplace_classifier.laplace_classifier import LaplaceInstrumentClassifier
import numpy as np

# Load model
classifier = LaplaceInstrumentClassifier()
classifier.load('$MODEL_OUTPUT')

# Test prediction
test_features = np.random.randn(26)
result = classifier.predict(test_features)

print(f'  Sample prediction: {result[\"label\"]} (confidence: {result[\"confidence\"]:.1%})')
print(f'  MIDI program: {result[\"midi_program\"]}')
print('✓ Model validation successful')
"

# =============================================================================
# FINAL SUMMARY
# =============================================================================

echo ""
echo "======================================================================"
echo "PHASE 2 SETUP COMPLETE"
echo "======================================================================"
echo ""
echo "📊 Summary:"
echo "  - NSynth dataset: $TRACK_COUNT samples"
echo "  - Features extracted: 26 Laplace features per sample"
echo "  - Models trained:"
echo "    • Random Forest: $RF_ACC"
echo "    • XGBoost: $XGB_ACC"
echo "  - Default model: $MODEL_OUTPUT"
echo ""
echo "📁 Files created:"
echo "  - $FEATURES_FILE"
echo "  - $RF_MODEL"
echo "  - $XGB_MODEL"
echo "  - $MODEL_OUTPUT"
echo ""
echo "📋 Logs saved to: $LOG_DIR"
echo ""
echo "🎯 Next steps:"
echo "  1. Integrate classifier into MR-MT3 pipeline"
echo "  2. Run comparison on validation tracks"
echo "  3. Evaluate improvements vs Phase 1 heuristics"
echo ""
echo "======================================================================"
