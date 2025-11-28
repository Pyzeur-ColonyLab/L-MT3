"""
Phase 1 MR-MT3 Laplace Enhancement Pipeline

Main integration script that orchestrates Laplace-based post-processing of MR-MT3
transcriptions. Applies decay-based consolidation and timbre-based refinement to
reduce onset fragmentation while preserving musical expressivity.

Usage:
    # As a module
    from phase1_mrmt3_enhancement import enhance_transcription
    midi_enhanced = enhance_transcription(midi, audio, sr=16000)

    # From command line
    python phase1_mrmt3_enhancement.py --midi input.mid --audio input.wav

Author: Laplace Research Team
Version: 1.0.0
"""

import os
import sys
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pretty_midi
import librosa

from metrics_recorder import MetricsRecorder

# Laplace enhancement modules
from laplace_mrmt3.config import EnhancementConfig
from laplace_mrmt3.features import extract_features_from_midi
from laplace_mrmt3.consolidation import consolidate_by_decay
from laplace_mrmt3.refinement import refine_by_timbre
from laplace_mrmt3.metrics import EnhancementMetrics, compare_transcriptions


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancementPipeline:
    """
    Main enhancement pipeline coordinating all Laplace processing stages.

    Attributes:
        config: EnhancementConfig instance with processing parameters
        metrics_tracker: EnhancementMetrics instance for evaluation
        verbose: Enable detailed logging if True
    """

    def __init__(self, config: Optional[EnhancementConfig] = None, verbose: bool = False):
        """
        Initialize enhancement pipeline.

        Args:
            config: EnhancementConfig instance (creates default if None)
            verbose: Enable verbose logging for debugging
        """
        self.config = config or EnhancementConfig()
        self.metrics_tracker = EnhancementMetrics()
        self.verbose = verbose

        if self.verbose:
            logger.setLevel(logging.DEBUG)
            logger.debug("Enhanced pipeline initialized in verbose mode")

    def validate_inputs(
        self,
        midi: pretty_midi.PrettyMIDI,
        audio: np.ndarray,
        sr: int
    ) -> None:
        """
        Validate input MIDI and audio data.

        Args:
            midi: PrettyMIDI object to validate
            audio: Audio waveform to validate
            sr: Sample rate to validate

        Raises:
            ValueError: If inputs are invalid or incompatible
        """
        # Validate MIDI
        if not isinstance(midi, pretty_midi.PrettyMIDI):
            raise ValueError("MIDI input must be a pretty_midi.PrettyMIDI object")

        total_notes = sum(len(inst.notes) for inst in midi.instruments)
        if total_notes == 0:
            raise ValueError("MIDI file contains no notes")

        # Validate audio
        if not isinstance(audio, np.ndarray):
            raise ValueError("Audio input must be a numpy array")

        if audio.ndim > 2:
            raise ValueError(f"Audio must be 1D or 2D, got {audio.ndim}D")

        if audio.ndim == 2:
            # Convert stereo to mono
            logger.warning("Converting stereo audio to mono")

        if sr <= 0:
            raise ValueError(f"Sample rate must be positive, got {sr}")

        # Validate alignment
        midi_duration = midi.get_end_time()
        audio_duration = len(audio) / sr if audio.ndim == 1 else len(audio[0]) / sr

        duration_diff = abs(midi_duration - audio_duration)
        if duration_diff > 1.0:  # Allow 1 second tolerance
            logger.warning(
                f"MIDI duration ({midi_duration:.2f}s) and audio duration "
                f"({audio_duration:.2f}s) differ by {duration_diff:.2f}s"
            )

    def enhance_transcription(
        self,
        midi: pretty_midi.PrettyMIDI,
        audio: np.ndarray,
        sr: int = 16000,
        return_intermediates: bool = False
    ) -> Tuple[pretty_midi.PrettyMIDI, Dict[str, Any]]:
        """
        Main enhancement pipeline: features → consolidation → refinement → metrics.

        Args:
            midi: Raw MR-MT3 transcription as PrettyMIDI object
            audio: Audio waveform (mono or stereo)
            sr: Sample rate of audio
            return_intermediates: Return intermediate MIDI states if True

        Returns:
            Tuple of (enhanced_midi, report_dict):
                - enhanced_midi: Post-processed PrettyMIDI object
                - report_dict: Comprehensive enhancement report with metrics

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If enhancement pipeline fails critically
        """
        logger.info("=" * 70)
        logger.info("Starting Phase 1 MR-MT3 Laplace Enhancement Pipeline")
        logger.info("=" * 70)

        start_time = time.time()

        # Validate inputs
        try:
            self.validate_inputs(midi, audio, sr)
        except ValueError as e:
            logger.error(f"Input validation failed: {e}")
            raise

        # Convert stereo to mono if needed
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)

        # Initialize report
        report = {
            'pipeline_version': '1.0.0',
            'config': self.config.to_dict(),
            'input_stats': {
                'total_notes': sum(len(inst.notes) for inst in midi.instruments),
                'total_instruments': len(midi.instruments),
                'duration_seconds': midi.get_end_time(),
                'audio_duration_seconds': len(audio) / sr,
                'sample_rate': sr
            },
            'stages': {},
            'timing': {}
        }

        if return_intermediates:
            report['intermediates'] = {}

        # Stage 1: Feature Extraction
        logger.info("\n[Stage 1/4] Extracting Laplace features from MIDI + audio...")
        stage_start = time.time()

        try:
            features = extract_features_from_midi(
                midi=midi,
                audio=audio,
                sr=sr,
                config=self.config
            )

            report['stages']['feature_extraction'] = {
                'status': 'success',
                'features_extracted': list(features.keys()),
                'prony_notes': len(features.get('decay_constants', {})),
                'timbre_notes': len(features.get('spectral_centroids', {}))
            }

            logger.info(f"✓ Extracted features for {len(features.get('decay_constants', {}))} notes")

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            report['stages']['feature_extraction'] = {
                'status': 'failed',
                'error': str(e)
            }

            # Graceful degradation: return original MIDI with error report
            if "Prony" in str(e) or "decay" in str(e):
                logger.warning("Prony analysis failed, returning original MIDI")
                report['pipeline_status'] = 'degraded'
                return midi, report
            else:
                raise RuntimeError(f"Critical feature extraction failure: {e}")

        report['timing']['feature_extraction'] = time.time() - stage_start

        # Stage 2: Decay-based Consolidation
        logger.info("\n[Stage 2/4] Applying decay-based onset consolidation...")
        stage_start = time.time()

        try:
            midi_consolidated, consolidation_stats = consolidate_by_decay(
                midi=midi,
                features=features,
                config=self.config
            )

            report['stages']['consolidation'] = {
                'status': 'success',
                'stats': consolidation_stats
            }

            logger.info(f"✓ Merged {consolidation_stats.get('total_merged', 0)} onset groups")

            if return_intermediates:
                report['intermediates']['post_consolidation'] = midi_consolidated

        except Exception as e:
            logger.error(f"Consolidation failed: {e}")
            report['stages']['consolidation'] = {
                'status': 'failed',
                'error': str(e)
            }
            # Continue with original MIDI
            midi_consolidated = midi

        report['timing']['consolidation'] = time.time() - stage_start

        # Stage 3: Timbre-based Refinement
        logger.info("\n[Stage 3/4] Applying timbre-based duration refinement...")
        stage_start = time.time()

        try:
            midi_enhanced, refinement_report = refine_by_timbre(
                midi=midi_consolidated,
                audio=audio,
                sr=sr,
                config=self.config,
                features=features
            )

            report['stages']['refinement'] = {
                'status': 'success',
                'report': refinement_report
            }

            # Count number of refinement decisions from report
            n_refinements = refinement_report.count("Instrument") if "Instrument" in refinement_report else 0
            logger.info(f"✓ Refinement complete")

        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            report['stages']['refinement'] = {
                'status': 'failed',
                'error': str(e)
            }
            # Use consolidated MIDI if refinement fails
            midi_enhanced = midi_consolidated

        report['timing']['refinement'] = time.time() - stage_start

        # Stage 4: Evaluation Metrics
        logger.info("\n[Stage 4/4] Computing enhancement metrics...")
        stage_start = time.time()

        try:
            # Compute basic improvement metrics
            metrics = {
                'instruments_original': len(midi.instruments),
                'instruments_enhanced': len(midi_enhanced.instruments),
                'reduction_count': len(midi.instruments) - len(midi_enhanced.instruments),
                'reduction_percent': 100 * (len(midi.instruments) - len(midi_enhanced.instruments)) / max(len(midi.instruments), 1),
                'notes_original': sum(len(inst.notes) for inst in midi.instruments),
                'notes_enhanced': sum(len(inst.notes) for inst in midi_enhanced.instruments)
            }

            report['stages']['metrics'] = {
                'status': 'success',
                'metrics': metrics
            }

            # Log key improvements
            if metrics['reduction_count'] > 0:
                logger.info(f"✓ Instruments reduced by {metrics['reduction_count']} ({metrics['reduction_percent']:.1f}%)")

        except Exception as e:
            logger.warning(f"Metrics computation failed: {e}")
            report['stages']['metrics'] = {
                'status': 'failed',
                'error': str(e)
            }

        report['timing']['metrics'] = time.time() - stage_start

        # Finalize report
        total_time = time.time() - start_time
        report['timing']['total_seconds'] = total_time
        report['pipeline_status'] = 'success'

        report['output_stats'] = {
            'total_notes': sum(len(inst.notes) for inst in midi_enhanced.instruments),
            'total_instruments': len(midi_enhanced.instruments),
            'duration_seconds': midi_enhanced.get_end_time()
        }

        logger.info("\n" + "=" * 70)
        logger.info(f"Enhancement Complete in {total_time:.2f}s")
        logger.info("=" * 70)

        return midi_enhanced, report

    def enhance_from_file(
        self,
        midi_path: str,
        audio_path: str,
        output_path: Optional[str] = None
    ) -> Tuple[pretty_midi.PrettyMIDI, Dict[str, Any]]:
        """
        Convenience function for file-based enhancement workflow.

        Args:
            midi_path: Path to input MIDI file
            audio_path: Path to input audio file (WAV, MP3, etc.)
            output_path: Path to save enhanced MIDI (auto-generated if None)

        Returns:
            Tuple of (enhanced_midi, report_dict)

        Raises:
            FileNotFoundError: If input files don't exist
            RuntimeError: If loading or saving fails
        """
        # Validate input paths
        if not os.path.exists(midi_path):
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Loading MIDI from: {midi_path}")
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load MIDI file: {e}")

        logger.info(f"Loading audio from: {audio_path}")
        try:
            audio, sr = librosa.load(audio_path, sr=self.config.sample_rate, mono=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file: {e}")

        # Run enhancement
        midi_enhanced, report = self.enhance_transcription(midi, audio, sr)

        # Save enhanced MIDI
        if output_path is None:
            base_name = Path(midi_path).stem
            output_path = str(Path(midi_path).parent / f"{base_name}_enhanced.mid")

        logger.info(f"Saving enhanced MIDI to: {output_path}")
        try:
            midi_enhanced.write(output_path)
            report['output_path'] = output_path
        except Exception as e:
            logger.error(f"Failed to save MIDI file: {e}")
            raise RuntimeError(f"Failed to save enhanced MIDI: {e}")

        # Record metrics if metrics_recorder is enabled
        if hasattr(self, 'metrics_recorder') and self.metrics_recorder is not None:
            try:
                # Extract track ID from path (e.g., Track00001)
                track_name = Path(midi_path).stem
                track_id = track_name.split('_')[0] if '_' in track_name else track_name

                # Get audio duration
                audio_duration = len(audio) / sr

                # Record metrics
                self.metrics_recorder.record_track_metrics(
                    track_id=track_id,
                    track_name=track_name,
                    report=report,
                    audio_duration=audio_duration
                )
                logger.info(f"✓ Metrics recorded for {track_id}")
            except Exception as e:
                logger.warning(f"Failed to record metrics: {e}")

        return midi_enhanced, report

    def compare_enhancement(
        self,
        midi_original: pretty_midi.PrettyMIDI,
        midi_enhanced: pretty_midi.PrettyMIDI,
        audio: np.ndarray,
        sr: int,
        ground_truth: Optional[pretty_midi.PrettyMIDI] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation comparing original and enhanced transcriptions.

        Args:
            midi_original: Original MR-MT3 transcription
            midi_enhanced: Laplace-enhanced transcription
            audio: Audio waveform
            sr: Sample rate
            ground_truth: Optional ground truth MIDI for reference

        Returns:
            Dictionary with detailed comparison metrics and statistical analysis
        """
        logger.info("Computing comprehensive enhancement comparison...")

        comparison = {
            'original': {},
            'enhanced': {},
            'improvement': {},
            'statistical_analysis': {}
        }

        # Basic statistics
        comparison['original']['total_notes'] = sum(len(inst.notes) for inst in midi_original.instruments)
        comparison['enhanced']['total_notes'] = sum(len(inst.notes) for inst in midi_enhanced.instruments)

        note_reduction = comparison['original']['total_notes'] - comparison['enhanced']['total_notes']
        comparison['improvement']['notes_reduced'] = note_reduction
        comparison['improvement']['reduction_percentage'] = (
            (note_reduction / comparison['original']['total_notes'] * 100)
            if comparison['original']['total_notes'] > 0 else 0
        )

        # Compute detailed metrics
        try:
            metrics = compare_transcriptions(
                midi_original=midi_original,
                midi_enhanced=midi_enhanced,
                audio=audio,
                sr=sr,
                config=self.config
            )
            comparison['metrics'] = metrics
        except Exception as e:
            logger.warning(f"Failed to compute detailed metrics: {e}")

        # Ground truth comparison if available
        if ground_truth is not None:
            logger.info("Computing ground truth comparison...")
            try:
                from laplace_mrmt3.metrics import compute_mir_eval_metrics

                gt_original = compute_mir_eval_metrics(midi_original, ground_truth)
                gt_enhanced = compute_mir_eval_metrics(midi_enhanced, ground_truth)

                comparison['ground_truth'] = {
                    'original': gt_original,
                    'enhanced': gt_enhanced,
                    'improvement': {
                        'f1_delta': gt_enhanced.get('f1', 0) - gt_original.get('f1', 0),
                        'precision_delta': gt_enhanced.get('precision', 0) - gt_original.get('precision', 0),
                        'recall_delta': gt_enhanced.get('recall', 0) - gt_original.get('recall', 0)
                    }
                }
            except Exception as e:
                logger.warning(f"Ground truth comparison failed: {e}")

        return comparison


# Public API functions

def enhance_transcription(
    midi: pretty_midi.PrettyMIDI,
    audio: np.ndarray,
    sr: int = 16000,
    config: Optional[EnhancementConfig] = None,
    verbose: bool = False
) -> Tuple[pretty_midi.PrettyMIDI, Dict[str, Any]]:
    """
    Main public API: Apply Laplace enhancement to MR-MT3 transcription.

    Args:
        midi: Raw MR-MT3 transcription as PrettyMIDI object
        audio: Audio waveform (mono or stereo)
        sr: Sample rate of audio
        config: Optional custom configuration (uses defaults if None)
        verbose: Enable verbose logging

    Returns:
        Tuple of (enhanced_midi, report_dict)

    Example:
        >>> from mr_mt3.inference import infer
        >>> from phase1_mrmt3_enhancement import enhance_transcription
        >>>
        >>> # Step 1: Run vanilla MR-MT3
        >>> audio = load_audio("song.wav", sr=16000)
        >>> midi_raw = infer(audio, model=checkpoint)
        >>>
        >>> # Step 2: Apply Laplace enhancement
        >>> midi_enhanced, report = enhance_transcription(midi_raw, audio, sr=16000)
        >>> print(f"Reduced notes by {report['improvement']['reduction_percentage']:.1f}%")
    """
    pipeline = EnhancementPipeline(config=config, verbose=verbose)
    return pipeline.enhance_transcription(midi, audio, sr)


def enhance_from_file(
    midi_path: str,
    audio_path: str,
    output_path: Optional[str] = None,
    config_path: Optional[str] = None,
    verbose: bool = False,
    metrics_dir: Optional[str] = None
) -> Tuple[pretty_midi.PrettyMIDI, Dict[str, Any]]:
    """
    Convenience function for file-based enhancement workflow.

    Args:
        midi_path: Path to input MIDI file
        audio_path: Path to input audio file
        output_path: Path to save enhanced MIDI (auto-generated if None)
        config_path: Path to YAML configuration file (uses defaults if None)
        verbose: Enable verbose logging
        metrics_dir: Directory to save metrics (enables metrics recording)

    Returns:
        Tuple of (enhanced_midi, report_dict)

    Example:
        >>> midi_enhanced, report = enhance_from_file(
        ...     midi_path="output_raw.mid",
        ...     audio_path="song.wav",
        ...     output_path="output_enhanced.mid",
        ...     metrics_dir="metrics"
        ... )
    """
    # Load config if provided
    config = None
    if config_path is not None:
        config = EnhancementConfig.from_yaml(config_path)

    pipeline = EnhancementPipeline(config=config, verbose=verbose)

    # Enable metrics recording if requested
    if metrics_dir is not None:
        pipeline.metrics_recorder = MetricsRecorder(output_dir=metrics_dir)
        logger.info(f"Metrics recording enabled: {metrics_dir}")

    return pipeline.enhance_from_file(midi_path, audio_path, output_path)


def compare_enhancement(
    midi_original: pretty_midi.PrettyMIDI,
    midi_enhanced: pretty_midi.PrettyMIDI,
    audio: np.ndarray,
    sr: int,
    ground_truth: Optional[pretty_midi.PrettyMIDI] = None,
    config: Optional[EnhancementConfig] = None
) -> Dict[str, Any]:
    """
    Comprehensive evaluation comparing original and enhanced transcriptions.

    Args:
        midi_original: Original MR-MT3 transcription
        midi_enhanced: Laplace-enhanced transcription
        audio: Audio waveform
        sr: Sample rate
        ground_truth: Optional ground truth MIDI for reference
        config: Optional custom configuration

    Returns:
        Dictionary with detailed comparison metrics
    """
    pipeline = EnhancementPipeline(config=config)
    return pipeline.compare_enhancement(
        midi_original, midi_enhanced, audio, sr, ground_truth
    )


def print_enhancement_report(report: Dict[str, Any]) -> None:
    """
    Pretty-print enhancement report to console.

    Args:
        report: Report dictionary from enhance_transcription()
    """
    print("\n" + "=" * 70)
    print("LAPLACE ENHANCEMENT REPORT")
    print("=" * 70)

    # Input stats
    print("\n[INPUT]")
    input_stats = report.get('input_stats', {})
    print(f"  Notes: {input_stats.get('total_notes', 'N/A')}")
    print(f"  Instruments: {input_stats.get('total_instruments', 'N/A')}")
    print(f"  Duration: {input_stats.get('duration_seconds', 0):.2f}s")

    # Stage results
    print("\n[PIPELINE STAGES]")
    for stage_name, stage_data in report.get('stages', {}).items():
        status = stage_data.get('status', 'unknown')
        symbol = "✓" if status == "success" else "✗"
        print(f"  {symbol} {stage_name.replace('_', ' ').title()}: {status}")

        if status == 'success':
            if stage_name == 'consolidation':
                stats = stage_data.get('stats', {})
                print(f"      → Merged {stats.get('total_merged', 0)} onset groups")
            elif stage_name == 'refinement':
                rep = stage_data.get('report', '')
                # Report is a string, extract refinement count if available
                if "Total refinements:" in rep:
                    # Extract number from "Total refinements: N"
                    import re
                    match = re.search(r'Total refinements:\s+(\d+)', rep)
                    if match:
                        n = match.group(1)
                        print(f"      → Applied {n} refinements")
                else:
                    print(f"      → Refinement applied")

    # Output stats
    print("\n[OUTPUT]")
    output_stats = report.get('output_stats', {})
    input_notes = input_stats.get('total_notes', 1)
    output_notes = output_stats.get('total_notes', 0)
    reduction = ((input_notes - output_notes) / input_notes * 100) if input_notes > 0 else 0
    print(f"  Notes: {output_notes} (reduced by {reduction:.1f}%)")
    print(f"  Duration: {output_stats.get('duration_seconds', 0):.2f}s")

    # Metrics
    if 'metrics' in report.get('stages', {}).get('metrics', {}):
        print("\n[METRICS]")
        metrics = report['stages']['metrics']['metrics']

        if 'onset_fragmentation' in metrics:
            frag = metrics['onset_fragmentation']
            print(f"  Onset Fragmentation: {frag.get('enhanced', 0):.3f} (was {frag.get('original', 0):.3f})")

        if 'harmonic_coherence' in metrics:
            coh = metrics['harmonic_coherence']
            print(f"  Harmonic Coherence: {coh.get('enhanced', 0):.3f} (was {coh.get('original', 0):.3f})")

    # Timing
    print("\n[PERFORMANCE]")
    timing = report.get('timing', {})
    print(f"  Total time: {timing.get('total_seconds', 0):.2f}s")
    print(f"  Feature extraction: {timing.get('feature_extraction', 0):.2f}s")
    print(f"  Consolidation: {timing.get('consolidation', 0):.2f}s")
    print(f"  Refinement: {timing.get('refinement', 0):.2f}s")

    print("\n" + "=" * 70 + "\n")


def main():
    """Command-line interface for enhancement pipeline."""
    parser = argparse.ArgumentParser(
        description="Phase 1 MR-MT3 Laplace Enhancement Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enhance a single file
  python phase1_mrmt3_enhancement.py --midi song.mid --audio song.wav

  # Use custom configuration
  python phase1_mrmt3_enhancement.py --midi song.mid --audio song.wav --config custom.yaml

  # Verbose output for debugging
  python phase1_mrmt3_enhancement.py --midi song.mid --audio song.wav --verbose
        """
    )

    parser.add_argument('--midi', type=str, required=True,
                       help='Path to input MIDI file (MR-MT3 output)')
    parser.add_argument('--audio', type=str, required=True,
                       help='Path to input audio file (WAV, MP3, etc.)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to output enhanced MIDI (auto-generated if not specified)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML configuration file (uses defaults if not specified)')
    parser.add_argument('--ground-truth', type=str, default=None,
                       help='Path to ground truth MIDI for evaluation')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--report-json', type=str, default=None,
                       help='Save JSON report to specified path')
    parser.add_argument('--metrics-dir', type=str, default=None,
                       help='Directory to save metrics (enables metrics recording)')

    args = parser.parse_args()

    # Run enhancement
    try:
        midi_enhanced, report = enhance_from_file(
            midi_path=args.midi,
            audio_path=args.audio,
            output_path=args.output,
            config_path=args.config,
            verbose=args.verbose,
            metrics_dir=args.metrics_dir
        )

        # Print report
        print_enhancement_report(report)

        # Ground truth comparison if provided
        if args.ground_truth:
            logger.info(f"Loading ground truth from: {args.ground_truth}")
            gt_midi = pretty_midi.PrettyMIDI(args.ground_truth)

            audio, sr = librosa.load(args.audio, sr=16000, mono=True)
            midi_original = pretty_midi.PrettyMIDI(args.midi)

            comparison = compare_enhancement(
                midi_original=midi_original,
                midi_enhanced=midi_enhanced,
                audio=audio,
                sr=sr,
                ground_truth=gt_midi
            )

            print("\n[GROUND TRUTH COMPARISON]")
            gt_data = comparison.get('ground_truth', {})
            if 'improvement' in gt_data:
                imp = gt_data['improvement']
                print(f"  F1 improvement: {imp.get('f1_delta', 0):+.3f}")
                print(f"  Precision improvement: {imp.get('precision_delta', 0):+.3f}")
                print(f"  Recall improvement: {imp.get('recall_delta', 0):+.3f}")

        # Save JSON report if requested
        if args.report_json:
            import json
            with open(args.report_json, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved JSON report to: {args.report_json}")

        logger.info("Enhancement completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Enhancement failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    # Demo: Load sample from babySlakh and run complete pipeline
    import sys

    # Check if running as demo or CLI
    if len(sys.argv) > 1:
        # Run CLI
        sys.exit(main())
    else:
        # Run demo
        print("=" * 70)
        print("PHASE 1 MR-MT3 LAPLACE ENHANCEMENT - DEMO")
        print("=" * 70)
        print("\nUsage: python phase1_mrmt3_enhancement.py --midi FILE.mid --audio FILE.wav")
        print("\nFor demo, provide paths via command-line arguments.")
        print("\nExample:")
        print("  python phase1_mrmt3_enhancement.py \\")
        print("    --midi data/babySlakh/Track00001_raw.mid \\")
        print("    --audio data/babySlakh/Track00001.wav \\")
        print("    --verbose")
        print("\n" + "=" * 70)
