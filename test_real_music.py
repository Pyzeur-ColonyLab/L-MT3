#!/usr/bin/env python3
"""
Real Music Testing Pipeline for L-MT3 Laplace Classifier

Comprehensive testing script that:
1. Processes multiple real music audio files
2. Runs MR-MT3 inference → ML-based enhancement
3. Exports per-instrument MIDI files with proper naming
4. Generates detailed reports for each track and summary report

Usage:
    python test_real_music.py --audio-dir ./music_samples --output-dir ./results --classifier ./laplace_classifier.pkl

    # Process single file
    python test_real_music.py --audio ./song.wav --output-dir ./results --classifier ./laplace_classifier.pkl

Author: Dyapason Research
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time
import traceback

import numpy as np
import pretty_midi
import librosa

# Import L-MT3 components
from phase1_mrmt3_enhancement import EnhancementPipeline
from laplace_mrmt3.config import EnhancementConfig
from run_mrmt3_inference import run_inference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealMusicTester:
    """
    Comprehensive testing system for real music with ML classifier
    """

    def __init__(
        self,
        classifier_path: str,
        model_path: Optional[str] = None,
        output_dir: str = './test_results',
        config: Optional[EnhancementConfig] = None
    ):
        """
        Initialize real music tester

        Args:
            classifier_path: Path to trained Laplace classifier (.pkl)
            model_path: Path to MR-MT3 model checkpoint (.pth) - if None, expects existing MIDI
            output_dir: Root output directory for all results
            config: Enhancement configuration (uses defaults if None)
        """
        self.classifier_path = Path(classifier_path)
        self.model_path = Path(model_path) if model_path else None
        self.output_dir = Path(output_dir)
        self.config = config or EnhancementConfig()

        # Validate classifier exists
        if not self.classifier_path.exists():
            raise FileNotFoundError(f"Classifier not found: {self.classifier_path}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize enhancement pipeline
        self.pipeline = EnhancementPipeline(
            config=self.config,
            verbose=True,
            use_ml_classifier=True,
            classifier_path=str(self.classifier_path)
        )

        # Track results
        self.results: List[Dict] = []

        logger.info(f"Initialized RealMusicTester")
        logger.info(f"  Classifier: {self.classifier_path}")
        logger.info(f"  Output dir: {self.output_dir}")

    def process_audio_file(
        self,
        audio_path: Path,
        baseline_midi_path: Optional[Path] = None
    ) -> Dict:
        """
        Process a single audio file through complete pipeline

        Args:
            audio_path: Path to input audio file
            baseline_midi_path: Optional path to existing baseline MIDI (skips MR-MT3 inference)

        Returns:
            Dictionary with processing results and paths
        """
        logger.info("=" * 80)
        logger.info(f"Processing: {audio_path.name}")
        logger.info("=" * 80)

        start_time = time.time()
        track_name = audio_path.stem

        # Create track-specific output directory
        track_output_dir = self.output_dir / track_name
        track_output_dir.mkdir(parents=True, exist_ok=True)

        result = {
            'track_name': track_name,
            'audio_path': str(audio_path),
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'error': None,
            'paths': {},
            'stats': {},
            'report': {}
        }

        try:
            # Step 1: Load audio
            logger.info("\n[1/5] Loading audio...")
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            audio_duration = len(audio) / sr
            logger.info(f"  Duration: {audio_duration:.2f}s")
            result['stats']['audio_duration'] = audio_duration

            # Step 2: MR-MT3 Inference (if baseline MIDI not provided)
            if baseline_midi_path is None:
                logger.info("\n[2/5] Running MR-MT3 inference...")

                if self.model_path is None:
                    raise ValueError("model_path required when baseline_midi_path not provided")

                baseline_midi_path = (track_output_dir / f"{track_name}_baseline.mid").resolve()

                inference_success = run_inference(
                    model_path=str(Path(self.model_path).resolve()),
                    audio_path=str(Path(audio_path).resolve()),
                    output_path=str(baseline_midi_path),
                    device='cuda'
                )

                if not inference_success:
                    raise RuntimeError("MR-MT3 inference failed")

                logger.info(f"  Baseline MIDI: {baseline_midi_path}")
            else:
                logger.info(f"\n[2/5] Using provided baseline MIDI: {baseline_midi_path}")

            result['paths']['baseline_midi'] = str(baseline_midi_path)

            # Load baseline MIDI
            baseline_midi = pretty_midi.PrettyMIDI(str(baseline_midi_path))
            result['stats']['baseline_instruments'] = len(baseline_midi.instruments)
            result['stats']['baseline_notes'] = sum(len(inst.notes) for inst in baseline_midi.instruments)

            # Step 3: ML-based Enhancement
            logger.info("\n[3/5] Running ML-based enhancement...")
            enhanced_midi_path = track_output_dir / f"{track_name}_enhanced.mid"

            enhanced_midi, enhancement_report = self.pipeline.enhance_transcription(
                midi=baseline_midi,
                audio=audio,
                sr=sr
            )

            # Save enhanced MIDI
            enhanced_midi.write(str(enhanced_midi_path))
            result['paths']['enhanced_midi'] = str(enhanced_midi_path)

            result['stats']['enhanced_instruments'] = len(enhanced_midi.instruments)
            result['stats']['enhanced_notes'] = sum(len(inst.notes) for inst in enhanced_midi.instruments)
            result['report'] = enhancement_report

            logger.info(f"  Enhanced MIDI: {enhanced_midi_path}")
            logger.info(f"  Instruments: {result['stats']['baseline_instruments']} → {result['stats']['enhanced_instruments']}")

            # Step 4: Export per-instrument MIDI files
            logger.info("\n[4/5] Exporting per-instrument MIDI files...")
            instrument_dir = track_output_dir / "instruments"
            instrument_dir.mkdir(exist_ok=True)

            instrument_paths = self._export_per_instrument_midi(
                midi=enhanced_midi,
                output_dir=instrument_dir,
                track_name=track_name
            )

            result['paths']['instruments'] = instrument_paths
            result['stats']['exported_instruments'] = len(instrument_paths)

            logger.info(f"  Exported {len(instrument_paths)} instrument files")

            # Step 5: Generate track report
            logger.info("\n[5/5] Generating track report...")
            report_path = track_output_dir / f"{track_name}_report.json"

            with open(report_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)

            result['paths']['report'] = str(report_path)
            logger.info(f"  Report: {report_path}")

            # Generate human-readable report
            text_report_path = track_output_dir / f"{track_name}_report.txt"
            self._generate_text_report(result, text_report_path)
            result['paths']['text_report'] = str(text_report_path)

            # Success
            result['success'] = True
            result['stats']['processing_time'] = time.time() - start_time

            logger.info(f"\n✓ Processing complete in {result['stats']['processing_time']:.2f}s")

        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            logger.error(f"✗ Processing failed: {e}")
            if logger.level == logging.DEBUG:
                logger.debug(traceback.format_exc())

        return result

    def _export_per_instrument_midi(
        self,
        midi: pretty_midi.PrettyMIDI,
        output_dir: Path,
        track_name: str
    ) -> List[str]:
        """
        Export each instrument to separate MIDI file with descriptive names

        Args:
            midi: PrettyMIDI object with multiple instruments
            output_dir: Directory to save instrument files
            track_name: Base track name for file naming

        Returns:
            List of paths to exported instrument MIDI files
        """
        exported_paths = []

        for idx, instrument in enumerate(midi.instruments):
            if len(instrument.notes) == 0:
                continue

            # Create instrument-specific MIDI
            inst_midi = pretty_midi.PrettyMIDI()
            inst_midi.instruments.append(instrument)

            # Generate descriptive filename
            program_name = pretty_midi.program_to_instrument_name(instrument.program)
            # Clean program name for filename
            clean_name = program_name.replace(' ', '_').replace('/', '_').lower()

            filename = f"{track_name}_{idx:02d}_{clean_name}_p{instrument.program}.mid"
            filepath = output_dir / filename

            # Save
            inst_midi.write(str(filepath))
            exported_paths.append(str(filepath))

            logger.info(f"    {idx:02d}: {program_name} (program {instrument.program}) - {len(instrument.notes)} notes")

        return exported_paths

    def _generate_text_report(self, result: Dict, output_path: Path):
        """
        Generate human-readable text report

        Args:
            result: Processing result dictionary
            output_path: Path to save text report
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"L-MT3 LAPLACE CLASSIFIER TEST REPORT")
        lines.append("=" * 80)
        lines.append(f"\nTrack: {result['track_name']}")
        lines.append(f"Timestamp: {result['timestamp']}")
        lines.append(f"Status: {'SUCCESS' if result['success'] else 'FAILED'}")

        if result.get('error'):
            lines.append(f"\nError: {result['error']}")

        lines.append(f"\n{'=' * 80}")
        lines.append("INPUT")
        lines.append("=" * 80)
        lines.append(f"Audio file: {result['audio_path']}")
        lines.append(f"Duration: {result['stats'].get('audio_duration', 0):.2f}s")

        lines.append(f"\n{'=' * 80}")
        lines.append("BASELINE TRANSCRIPTION (MR-MT3)")
        lines.append("=" * 80)
        lines.append(f"Instruments: {result['stats'].get('baseline_instruments', 0)}")
        lines.append(f"Total notes: {result['stats'].get('baseline_notes', 0)}")
        lines.append(f"MIDI file: {result['paths'].get('baseline_midi', 'N/A')}")

        lines.append(f"\n{'=' * 80}")
        lines.append("ML-ENHANCED TRANSCRIPTION (Phase 2)")
        lines.append("=" * 80)
        lines.append(f"Instruments: {result['stats'].get('enhanced_instruments', 0)}")
        lines.append(f"Total notes: {result['stats'].get('enhanced_notes', 0)}")
        lines.append(f"MIDI file: {result['paths'].get('enhanced_midi', 'N/A')}")

        # Calculate improvements
        baseline_inst = result['stats'].get('baseline_instruments', 0)
        enhanced_inst = result['stats'].get('enhanced_instruments', 0)
        inst_reduction = baseline_inst - enhanced_inst
        inst_reduction_pct = (inst_reduction / baseline_inst * 100) if baseline_inst > 0 else 0

        lines.append(f"\nImprovement:")
        lines.append(f"  Instrument reduction: {inst_reduction} ({inst_reduction_pct:.1f}%)")

        # ML Refinement stats
        if 'refinement' in result.get('report', {}).get('stages', {}):
            refinement = result['report']['stages']['refinement']
            lines.append(f"\nML Refinement:")
            lines.append(f"  Method: {refinement.get('method', 'N/A')}")
            if 'report' in refinement:
                lines.append(f"  Details:\n{refinement['report']}")

        lines.append(f"\n{'=' * 80}")
        lines.append("PER-INSTRUMENT MIDI FILES")
        lines.append("=" * 80)
        lines.append(f"Exported: {result['stats'].get('exported_instruments', 0)} files")

        if 'instruments' in result['paths']:
            for path in result['paths']['instruments']:
                filename = Path(path).name
                lines.append(f"  - {filename}")

        lines.append(f"\n{'=' * 80}")
        lines.append("PERFORMANCE")
        lines.append("=" * 80)

        if 'timing' in result.get('report', {}):
            timing = result['report']['timing']
            lines.append(f"Feature extraction: {timing.get('feature_extraction', 0):.2f}s")
            lines.append(f"Consolidation: {timing.get('consolidation', 0):.2f}s")
            lines.append(f"Refinement: {timing.get('refinement', 0):.2f}s")
            lines.append(f"Total: {timing.get('total_seconds', 0):.2f}s")

        lines.append(f"\nOverall processing time: {result['stats'].get('processing_time', 0):.2f}s")

        lines.append(f"\n{'=' * 80}")
        lines.append("FILES GENERATED")
        lines.append("=" * 80)
        for key, path in result['paths'].items():
            if isinstance(path, str):
                lines.append(f"{key}: {path}")

        lines.append("\n" + "=" * 80 + "\n")

        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

    def process_directory(
        self,
        audio_dir: Path,
        audio_extensions: Tuple[str, ...] = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
    ) -> List[Dict]:
        """
        Process all audio files in a directory

        Args:
            audio_dir: Directory containing audio files
            audio_extensions: Tuple of valid audio file extensions

        Returns:
            List of result dictionaries for all processed tracks
        """
        # Find all audio files
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(audio_dir.glob(f"*{ext}"))

        audio_files = sorted(audio_files)

        if not audio_files:
            logger.warning(f"No audio files found in {audio_dir}")
            return []

        logger.info(f"\nFound {len(audio_files)} audio files to process")

        # Process each file
        results = []
        for i, audio_path in enumerate(audio_files, 1):
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Track {i}/{len(audio_files)}")
            logger.info(f"{'=' * 80}")

            result = self.process_audio_file(audio_path)
            results.append(result)
            self.results.append(result)

        return results

    def generate_summary_report(self) -> Dict:
        """
        Generate summary report for all processed tracks

        Returns:
            Summary statistics dictionary
        """
        if not self.results:
            logger.warning("No results to summarize")
            return {}

        summary = {
            'total_tracks': len(self.results),
            'successful_tracks': sum(1 for r in self.results if r['success']),
            'failed_tracks': sum(1 for r in self.results if not r['success']),
            'total_processing_time': sum(r['stats'].get('processing_time', 0) for r in self.results),
            'average_processing_time': 0,
            'statistics': {
                'baseline_instruments': [],
                'enhanced_instruments': [],
                'baseline_notes': [],
                'enhanced_notes': [],
                'instrument_reduction': [],
                'processing_times': []
            },
            'tracks': []
        }

        # Collect statistics
        for result in self.results:
            if result['success']:
                stats = result['stats']
                summary['statistics']['baseline_instruments'].append(stats.get('baseline_instruments', 0))
                summary['statistics']['enhanced_instruments'].append(stats.get('enhanced_instruments', 0))
                summary['statistics']['baseline_notes'].append(stats.get('baseline_notes', 0))
                summary['statistics']['enhanced_notes'].append(stats.get('enhanced_notes', 0))

                baseline = stats.get('baseline_instruments', 0)
                enhanced = stats.get('enhanced_instruments', 0)
                reduction = baseline - enhanced
                summary['statistics']['instrument_reduction'].append(reduction)
                summary['statistics']['processing_times'].append(stats.get('processing_time', 0))

            summary['tracks'].append({
                'name': result['track_name'],
                'success': result['success'],
                'error': result.get('error'),
                'paths': result.get('paths', {})
            })

        # Calculate averages
        if summary['successful_tracks'] > 0:
            summary['average_processing_time'] = summary['total_processing_time'] / summary['successful_tracks']

            stats = summary['statistics']
            summary['averages'] = {
                'baseline_instruments': np.mean(stats['baseline_instruments']),
                'enhanced_instruments': np.mean(stats['enhanced_instruments']),
                'instrument_reduction': np.mean(stats['instrument_reduction']),
                'processing_time': np.mean(stats['processing_times'])
            }

        # Save summary
        summary_path = self.output_dir / 'summary_report.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"\nSummary report saved: {summary_path}")

        # Generate text summary
        text_summary_path = self.output_dir / 'summary_report.txt'
        self._generate_text_summary(summary, text_summary_path)

        return summary

    def _generate_text_summary(self, summary: Dict, output_path: Path):
        """Generate human-readable summary report"""
        lines = []
        lines.append("=" * 80)
        lines.append("L-MT3 LAPLACE CLASSIFIER - BATCH TESTING SUMMARY")
        lines.append("=" * 80)
        lines.append(f"\nTotal tracks processed: {summary['total_tracks']}")
        lines.append(f"Successful: {summary['successful_tracks']}")
        lines.append(f"Failed: {summary['failed_tracks']}")
        lines.append(f"Success rate: {summary['successful_tracks'] / summary['total_tracks'] * 100:.1f}%")

        lines.append(f"\n{'=' * 80}")
        lines.append("PROCESSING TIME")
        lines.append("=" * 80)
        lines.append(f"Total: {summary['total_processing_time']:.2f}s")
        lines.append(f"Average per track: {summary['average_processing_time']:.2f}s")

        if 'averages' in summary:
            avg = summary['averages']
            lines.append(f"\n{'=' * 80}")
            lines.append("AVERAGE STATISTICS")
            lines.append("=" * 80)
            lines.append(f"Baseline instruments: {avg['baseline_instruments']:.1f}")
            lines.append(f"Enhanced instruments: {avg['enhanced_instruments']:.1f}")
            lines.append(f"Instrument reduction: {avg['instrument_reduction']:.1f}")

        lines.append(f"\n{'=' * 80}")
        lines.append("INDIVIDUAL TRACKS")
        lines.append("=" * 80)

        for track in summary['tracks']:
            status = "✓" if track['success'] else "✗"
            lines.append(f"\n{status} {track['name']}")
            if not track['success']:
                lines.append(f"   Error: {track.get('error', 'Unknown')}")
            else:
                if 'enhanced_midi' in track.get('paths', {}):
                    lines.append(f"   Enhanced MIDI: {track['paths']['enhanced_midi']}")

        lines.append("\n" + "=" * 80 + "\n")

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        logger.info(f"Text summary saved: {output_path}")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Test L-MT3 Laplace Classifier on real music files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process directory of audio files
  python test_real_music.py --audio-dir ./music_samples --output-dir ./results --classifier ./laplace_classifier.pkl --model ./mr_mt3.pth

  # Process single file (with existing baseline MIDI)
  python test_real_music.py --audio ./song.wav --baseline-midi ./song_baseline.mid --output-dir ./results --classifier ./laplace_classifier.pkl
        """
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--audio', type=str, help='Path to single audio file')
    input_group.add_argument('--audio-dir', type=str, help='Directory containing audio files')

    # Required arguments
    parser.add_argument('--classifier', type=str, required=True,
                       help='Path to trained Laplace classifier (.pkl)')
    parser.add_argument('--output-dir', type=str, default='./test_results',
                       help='Output directory for results (default: ./test_results)')

    # Optional arguments
    parser.add_argument('--model', type=str, default=None,
                       help='Path to MR-MT3 model checkpoint (.pth) - required if baseline MIDI not provided')
    parser.add_argument('--baseline-midi', type=str, default=None,
                       help='Path to baseline MIDI (skips MR-MT3 inference if provided)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom enhancement config YAML')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config if provided
    config = None
    if args.config:
        config = EnhancementConfig.from_yaml(args.config)

    # Initialize tester
    try:
        tester = RealMusicTester(
            classifier_path=args.classifier,
            model_path=args.model,
            output_dir=args.output_dir,
            config=config
        )
    except Exception as e:
        logger.error(f"Failed to initialize tester: {e}")
        return 1

    # Process files
    try:
        if args.audio:
            # Single file
            audio_path = Path(args.audio)
            baseline_midi_path = Path(args.baseline_midi) if args.baseline_midi else None

            result = tester.process_audio_file(audio_path, baseline_midi_path)
            tester.results.append(result)

        elif args.audio_dir:
            # Directory of files
            audio_dir = Path(args.audio_dir)
            tester.process_directory(audio_dir)

        # Generate summary
        logger.info("\n" + "=" * 80)
        logger.info("Generating summary report...")
        logger.info("=" * 80)

        summary = tester.generate_summary_report()

        logger.info("\n" + "=" * 80)
        logger.info("TESTING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Results directory: {tester.output_dir}")
        logger.info(f"Processed: {summary['total_tracks']} tracks")
        logger.info(f"Success: {summary['successful_tracks']}/{summary['total_tracks']}")

        return 0 if summary['failed_tracks'] == 0 else 1

    except Exception as e:
        logger.error(f"Testing failed: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
