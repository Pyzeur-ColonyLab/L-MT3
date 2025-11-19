#!/usr/bin/env python3
"""
MR-MT3 Inference Wrapper
Command-line interface for running MR-MT3 inference on audio files.
"""

import argparse
import sys
from pathlib import Path
import os

# Add MR-MT3 to Python path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
MRMT3_DIR = PROJECT_ROOT / "mr-mt3"
sys.path.insert(0, str(MRMT3_DIR))

try:
    from inference import InferenceHandler
except ImportError as e:
    print(f"Error: Could not import MR-MT3. Make sure mr-mt3 directory exists.")
    print(f"Expected path: {MRMT3_DIR}")
    print(f"Error: {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run MR-MT3 inference on audio files"
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        required=True,
        help="Path to input audio file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save output MIDI files",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to MR-MT3 model checkpoint",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use (default: cpu)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get output filename
    audio_path = Path(args.audio_file)
    output_filename = audio_path.stem + ".mid"
    output_path = output_dir / output_filename

    if args.verbose:
        print(f"Input audio: {args.audio_file}")
        print(f"Output MIDI: {output_path}")
        print(f"Model: {args.model_path}")
        print(f"Device: {args.device}")

    # Initialize inference handler
    try:
        handler = InferenceHandler(
            weight_path=args.model_path,
            device=args.device,
        )
        if args.verbose:
            print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Run inference
    try:
        if args.verbose:
            print(f"Processing {audio_path.name}...")

        handler.inference(
            audio_path=str(args.audio_file),
            outpath=str(output_path),
            batch_size=args.batch_size,
            verbose=args.verbose,
        )

        if output_path.exists():
            print(f"Success: {output_filename}")
        else:
            print(f"Warning: Output file not created: {output_path}")
            sys.exit(1)

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
