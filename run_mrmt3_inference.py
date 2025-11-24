#!/usr/bin/env python3
"""
MR-MT3 Inference Wrapper for Local GPU Execution (PyTorch)

This script provides a command-line interface for running MR-MT3 transcription
locally on GPU instances using the PyTorch-based gudgud96/MR-MT3 implementation.

Usage:
    python run_mrmt3_inference.py --model MODEL_PATH --audio AUDIO_PATH --output OUTPUT_PATH

Requirements:
    - PyTorch with CUDA support
    - gudgud96/MR-MT3 repository cloned
    - Model checkpoint (.pth file)
"""

import argparse
import sys
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_inference(model_path: str, audio_path: str, output_path: str, device: str = 'cuda'):
    """
    Run MR-MT3 inference on audio file using PyTorch

    Args:
        model_path: Path to MR-MT3 .pth checkpoint
        audio_path: Path to input audio file
        output_path: Path to save output MIDI file
        device: Device to use ('cuda' or 'cpu')
    """
    try:
        # Import PyTorch and check GPU availability
        import torch
        import librosa
        import numpy as np

        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = 'cpu'
        elif device == 'cuda':
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Validate inputs
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            return False

        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return False

        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Processing audio: {audio_path}")

        # Get MR-MT3 repository path and patches
        model_dir = Path(model_path).parent
        mr_mt3_repo = model_dir / "MR-MT3"

        # Get script directory for patches
        script_dir = Path(__file__).parent
        patches_dir = script_dir / "mr_mt3_patches"

        # Prefer patched inference over repository version
        if patches_dir.exists():
            logger.info("Using patched MR-MT3 inference module...")
            sys.path.insert(0, str(patches_dir))
        elif mr_mt3_repo.exists():
            logger.info("Using MR-MT3 repository inference module...")
            sys.path.insert(0, str(mr_mt3_repo))
        else:
            logger.error(f"Neither patches ({patches_dir}) nor MR-MT3 repo ({mr_mt3_repo}) found")
            logger.error("Please ensure MR-MT3 is properly set up")
            return False

        # Also add MR-MT3 repo to path for dependencies
        if mr_mt3_repo.exists():
            sys.path.insert(0, str(mr_mt3_repo))

        # Import MR-MT3 inference handler
        logger.info("Loading InferenceHandler...")

        try:
            from inference import InferenceHandler

            # Initialize handler
            logger.info("Initializing InferenceHandler...")
            handler = InferenceHandler(
                weight_path=model_path,
                device=torch.device(device)
            )

            # Load and preprocess audio
            logger.info(f"Loading audio at 16kHz mono...")
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)

            # Run inference
            logger.info("Running MR-MT3 inference...")
            handler.inference(audio, audio_path, outpath=output_path)

            # Verify output
            if not os.path.exists(output_path):
                logger.error(f"MIDI file was not created at {output_path}")
                return False

            logger.info(f"Transcription saved to: {output_path}")
            return True

        except ImportError as e:
            logger.error(f"Failed to import MR-MT3 inference module: {e}")
            logger.error("The MR-MT3 repository may be missing or corrupted")
            return False

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Run MR-MT3 inference for music transcription (PyTorch)'
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Path to MR-MT3 .pth checkpoint file'
    )
    parser.add_argument(
        '--audio',
        required=True,
        help='Path to input audio file'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to save output MIDI file'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for inference (default: cuda)'
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("MR-MT3 Inference (PyTorch)")
    logger.info("=" * 70)

    success = run_inference(
        model_path=args.model,
        audio_path=args.audio,
        output_path=args.output,
        device=args.device
    )

    if success:
        logger.info("Inference completed successfully")
        sys.exit(0)
    else:
        logger.error("Inference failed")
        sys.exit(1)

if __name__ == '__main__':
    main()
