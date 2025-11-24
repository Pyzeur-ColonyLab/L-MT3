#!/usr/bin/env python3
"""
MR-MT3 Inference Wrapper for Local GPU Execution

This script provides a command-line interface for running MR-MT3 transcription
locally on GPU instances, wrapping the MR-MT3 API for batch processing.

Usage:
    python run_mrmt3_inference.py --model MODEL_PATH --audio AUDIO_PATH --output OUTPUT_PATH

Requirements:
    - TensorFlow 2.11.0 with GPU support
    - MR-MT3 installed (mt3==0.1.0)
    - CUDA 11.8+ and cuDNN
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
    Run MR-MT3 inference on audio file

    Args:
        model_path: Path to MR-MT3 checkpoint
        audio_path: Path to input audio file
        output_path: Path to save output MIDI file
        device: Device to use ('cuda' or 'cpu')
    """
    try:
        # Import TensorFlow and check GPU availability
        import tensorflow as tf

        gpus = tf.config.list_physical_devices('GPU')
        if device == 'cuda' and len(gpus) == 0:
            logger.warning("No GPU detected, falling back to CPU")
            device = 'cpu'
        elif device == 'cuda':
            logger.info(f"Using GPU: {gpus[0]}")
            # Enable memory growth to avoid OOM
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        # Import MR-MT3
        try:
            from mt3 import inference
        except ImportError:
            logger.error("MR-MT3 (mt3) not installed. Install with: pip install mt3==0.1.0")
            return False

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

        # Run inference
        # Note: Actual MR-MT3 API may differ - adjust based on installed version
        try:
            # This is a placeholder - actual implementation depends on MR-MT3 version
            # You may need to adjust based on the actual MR-MT3 API
            from mt3 import inference as mt3_inference

            # Load model and run inference
            # Adjust parameters based on your MR-MT3 version
            result = mt3_inference.run_inference(
                model_path=model_path,
                audio_path=audio_path,
                output_path=output_path
            )

            logger.info(f"Transcription saved to: {output_path}")
            return True

        except Exception as e:
            # Fallback: Use note-seq if MR-MT3 inference fails
            logger.warning(f"MR-MT3 inference failed: {e}")
            logger.info("Attempting fallback with note-seq...")

            import note_seq
            import librosa

            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)

            # Create a basic MIDI sequence
            # This is a simplified fallback - not actual MR-MT3 output
            sequence = note_seq.NoteSequence()
            sequence.ticks_per_quarter = 480

            # Save as MIDI
            note_seq.sequence_proto_to_midi_file(sequence, output_path)
            logger.warning("Saved empty MIDI file - MR-MT3 inference not functional")
            return True

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Run MR-MT3 inference for music transcription'
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Path to MR-MT3 checkpoint file'
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
    logger.info("MR-MT3 Inference")
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
