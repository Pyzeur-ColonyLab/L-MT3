#!/bin/bash
################################################################################
# Download MR-MT3 Pretrained Model from HuggingFace
#
# Downloads the MR-MT3 checkpoint from HuggingFace to the mr-mt3/pretrained/
# directory. This is needed because Git LFS fails to download the large model
# file during repository cloning.
#
# Usage:
#   ./download_mrmt3_model.sh [OUTPUT_DIR]
#
# Arguments:
#   OUTPUT_DIR  - Optional: Directory to save model (default: ../mr-mt3/pretrained)
################################################################################

set -euo pipefail

# Default output directory
DEFAULT_DIR="$(dirname "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")")/mr-mt3/pretrained"
OUTPUT_DIR="${1:-$DEFAULT_DIR}"

# HuggingFace model URL
# Based on: https://huggingface.co/gudgud1014/MR-MT3/tree/main
# Using slakh_f1_0.65.pth (proven working model from music-to-midi-api)
MODEL_URL="https://huggingface.co/gudgud1014/MR-MT3/resolve/main/slakh_f1_0.65.pth"
MODEL_FILE="$OUTPUT_DIR/mt3.pth"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[!]${NC} $1"; }

log_info "MR-MT3 Model Download Script"
log_info "Target directory: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if model already exists
if [ -f "$MODEL_FILE" ]; then
    SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    log_warning "Model already exists: $MODEL_FILE ($SIZE)"
    read -p "Overwrite? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Keeping existing model"
        exit 0
    fi
    rm "$MODEL_FILE"
fi

# Download model
log_info "Downloading MR-MT3 checkpoint from HuggingFace..."
log_info "URL: $MODEL_URL"
log_info "This may take several minutes (~400 MB)..."

# Try wget first, fall back to curl
if command -v wget &> /dev/null; then
    wget --progress=bar:force:noscroll \
         --show-progress \
         -O "$MODEL_FILE" \
         "$MODEL_URL"
elif command -v curl &> /dev/null; then
    curl -L \
         --progress-bar \
         -o "$MODEL_FILE" \
         "$MODEL_URL"
else
    echo "Error: Neither wget nor curl found. Install one of them:"
    echo "  macOS: brew install wget"
    echo "  Linux: sudo apt-get install wget"
    exit 1
fi

# Verify download
if [ -f "$MODEL_FILE" ]; then
    SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    log_success "Model downloaded successfully: $SIZE"
    log_success "Saved to: $MODEL_FILE"

    # Show file info
    echo ""
    echo "Model file details:"
    ls -lh "$MODEL_FILE"
else
    log_warning "Download may have failed - file not found"
    exit 1
fi

echo ""
log_success "MR-MT3 model ready for use!"
