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

        # Attempt to use MR-MT3 repository inference
        try:
            # Get MR-MT3 repository path
            model_dir = Path(model_path).parent
            mr_mt3_repo = model_dir / "MR-MT3"

            if not mr_mt3_repo.exists():
                raise ImportError(f"MR-MT3 repository not found at {mr_mt3_repo}")

            # Add MR-MT3 to Python path
            sys.path.insert(0, str(mr_mt3_repo))

            # Import MR-MT3 modules
            logger.info("Attempting to load MR-MT3 modules...")

            # Try to import inference module from MR-MT3
            try:
                from inference import inference_single_file

                # Run inference using MR-MT3's API
                logger.info("Running MR-MT3 inference...")
                inference_single_file(
                    model_path=model_path,
                    audio_path=audio_path,
                    output_path=output_path,
                    device=device
                )

                logger.info(f"Transcription saved to: {output_path}")
                return True

            except ImportError as e:
                logger.warning(f"Could not import MR-MT3 inference module: {e}")
                raise

        except Exception as e:
            logger.warning(f"MR-MT3 inference failed: {e}")
            logger.info("Attempting fallback with basic-pitch...")

            # Fallback: Use basic-pitch if available
            try:
                from basic_pitch.inference import predict
                from basic_pitch import ICASSP_2022_MODEL_PATH
                import pretty_midi

                logger.info("Using basic-pitch for transcription...")

                # Run basic-pitch
                model_output, midi_data, note_events = predict(
                    audio_path,
                    ICASSP_2022_MODEL_PATH
                )

                # Save MIDI
                midi_data.write(output_path)
                logger.warning(f"Used basic-pitch fallback - saved to: {output_path}")
                return True

            except ImportError:
                logger.error("basic-pitch not available for fallback")

            # Last resort: Create empty MIDI
            logger.warning("Creating empty MIDI file as last resort...")
            try:
                import pretty_midi

                # Create empty MIDI sequence
                midi = pretty_midi.PrettyMIDI()

                # Add a single piano instrument with no notes
                # This ensures the file is valid but clearly shows no transcription occurred
                piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
                piano = pretty_midi.Instrument(program=piano_program)
                midi.instruments.append(piano)

                # Save empty MIDI
                midi.write(output_path)
                logger.error(f"Saved empty MIDI file - MR-MT3 inference not functional: {output_path}")
                logger.error("Please ensure MR-MT3 repository has proper inference module")
                return False

            except Exception as e:
                logger.error(f"Failed to create fallback MIDI: {e}")
                return False

    except Exception as e:
        logger.error(f"Inference failed: {e}")
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
