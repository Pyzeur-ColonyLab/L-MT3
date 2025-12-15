# -*- coding: utf-8 -*-
"""
Source Separation Interface for L-MT3 Pipeline

Provides a flexible interface for integrating external source separation tools
(Demucs, Spleeter, Open-Unmix, etc.) into the Laplace refinement pipeline.

This module defines the interface and provides adapters for common tools.
Users can implement custom adapters for their preferred separation method.

Author: Dyapason Research
Date: 2024-12-12
"""

import numpy as np
import logging
from typing import Dict, Optional, List, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
import tempfile
import subprocess

logger = logging.getLogger(__name__)


# =============================================================================
# STEM TYPES
# =============================================================================

class StemType:
    """Standard stem types for source separation"""
    BASS = 'bass'
    DRUMS = 'drums'
    VOCALS = 'vocals'
    OTHER = 'other'
    PIANO = 'piano'
    GUITAR = 'guitar'

    # Mapping to MIDI General Instruments
    STEM_TO_MIDI_PROGRAMS = {
        BASS: list(range(32, 40)),       # Acoustic Bass (32-39)
        DRUMS: [128],                     # Percussion channel
        VOCALS: list(range(52, 56)),     # Choir Aahs, Voice Oohs, Synth Voice, Orchestra Hit
        PIANO: list(range(0, 8)),        # Acoustic Grand Piano to Clavinet (0-7)
        GUITAR: list(range(24, 32)),     # Acoustic Guitar to Electric Guitar (24-31)
        OTHER: None                       # Default fallback
    }

    @classmethod
    def get_stem_for_program(cls, program: int) -> str:
        """
        Map MIDI program number to stem type

        Args:
            program: MIDI program number (0-127, 128 for drums)

        Returns:
            Stem type identifier
        """
        if program == 128:
            return cls.DRUMS

        for stem_type, programs in cls.STEM_TO_MIDI_PROGRAMS.items():
            if programs and program in programs:
                return stem_type

        return cls.OTHER


# =============================================================================
# ABSTRACT INTERFACE
# =============================================================================

class SourceSeparator(ABC):
    """
    Abstract interface for source separation tools.

    Implement this interface to integrate your preferred separation tool
    (Demucs, Spleeter, Open-Unmix, etc.)
    """

    @abstractmethod
    def separate(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Separate audio into stems

        Args:
            audio: Audio waveform (mono or stereo)
            sr: Sample rate

        Returns:
            Dictionary mapping stem types to audio arrays
            e.g., {'bass': array, 'drums': array, 'vocals': array, 'other': array}
        """
        pass

    @property
    @abstractmethod
    def available_stems(self) -> List[str]:
        """List of stem types this separator can produce"""
        pass


# =============================================================================
# MOCK SEPARATOR (FOR TESTING)
# =============================================================================

class MockSeparator(SourceSeparator):
    """
    Mock separator for testing without external dependencies.

    Returns the original audio for each stem (no actual separation).
    """

    def separate(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Return original audio for all stems"""
        logger.warning("Using MockSeparator - no actual separation performed")

        stems = {}
        for stem_type in self.available_stems:
            stems[stem_type] = audio.copy()

        return stems

    @property
    def available_stems(self) -> List[str]:
        return [StemType.BASS, StemType.DRUMS, StemType.VOCALS, StemType.OTHER]


# =============================================================================
# DEMUCS ADAPTER
# =============================================================================

class DemucsSeparator(SourceSeparator):
    """
    Adapter for Demucs (https://github.com/facebookresearch/demucs)

    Installation:
        pip install demucs

    Usage:
        separator = DemucsSeparator(model='htdemucs')
        stems = separator.separate(audio, sr)
    """

    def __init__(self, model: str = 'htdemucs', device: str = 'cuda'):
        """
        Initialize Demucs separator

        Args:
            model: Demucs model name ('htdemucs', 'htdemucs_ft', 'mdx_extra')
            device: Device to use ('cuda' or 'cpu')
        """
        try:
            import torch
            from demucs.pretrained import get_model
            from demucs.apply import apply_model

            self.torch = torch
            self.apply_model = apply_model

            # Load model
            self.model = get_model(model)
            self.model.to(device)
            self.model.eval()
            self.device = device

            logger.info(f"Loaded Demucs model: {model} on {device}")

        except ImportError:
            raise ImportError(
                "Demucs not installed. Install with: pip install demucs"
            )

    def separate(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Separate audio using Demucs

        Args:
            audio: Audio waveform (mono or stereo)
            sr: Sample rate

        Returns:
            Dictionary with separated stems
        """
        # Ensure stereo
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)
        elif audio.ndim == 2 and audio.shape[0] > 2:
            # Transpose if channels are second dimension
            audio = audio.T

        # Convert to torch tensor
        audio_tensor = self.torch.from_numpy(audio).float().to(self.device)
        audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension

        # Apply model
        with self.torch.no_grad():
            sources = self.apply_model(
                self.model,
                audio_tensor,
                device=self.device,
                progress=False
            )[0]  # Remove batch dimension

        # Convert back to numpy
        stems = {}
        stem_names = self.model.sources

        for i, stem_name in enumerate(stem_names):
            stem_audio = sources[i].cpu().numpy()
            # Convert stereo to mono
            if stem_audio.shape[0] == 2:
                stem_audio = np.mean(stem_audio, axis=0)
            stems[stem_name] = stem_audio

        logger.info(f"Separated audio into {len(stems)} stems: {list(stems.keys())}")

        return stems

    @property
    def available_stems(self) -> List[str]:
        return ['drums', 'bass', 'other', 'vocals']


# =============================================================================
# SPLEETER ADAPTER
# =============================================================================

class SpleeterSeparator(SourceSeparator):
    """
    Adapter for Spleeter (https://github.com/deezer/spleeter)

    Installation:
        pip install spleeter

    Usage:
        separator = SpleeterSeparator(stems=4)
        stems = separator.separate(audio, sr)
    """

    def __init__(self, stems: int = 4):
        """
        Initialize Spleeter separator

        Args:
            stems: Number of stems (2, 4, or 5)
                2: vocals, accompaniment
                4: vocals, drums, bass, other
                5: vocals, drums, bass, piano, other
        """
        if stems not in [2, 4, 5]:
            raise ValueError("Spleeter stems must be 2, 4, or 5")

        try:
            from spleeter.separator import Separator

            self.separator = Separator(f'spleeter:{stems}stems')
            self.n_stems = stems

            logger.info(f"Loaded Spleeter with {stems} stems")

        except ImportError:
            raise ImportError(
                "Spleeter not installed. Install with: pip install spleeter"
            )

    def separate(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Separate audio using Spleeter

        Args:
            audio: Audio waveform (mono or stereo)
            sr: Sample rate

        Returns:
            Dictionary with separated stems
        """
        # Spleeter expects stereo
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=1)

        # Separate
        prediction = self.separator.separate(audio)

        # Convert to mono and normalize keys
        stems = {}
        for stem_name, stem_audio in prediction.items():
            # Convert stereo to mono
            if stem_audio.ndim == 2 and stem_audio.shape[1] == 2:
                stem_audio = np.mean(stem_audio, axis=1)
            stems[stem_name] = stem_audio

        logger.info(f"Separated audio into {len(stems)} stems: {list(stems.keys())}")

        return stems

    @property
    def available_stems(self) -> List[str]:
        if self.n_stems == 2:
            return ['vocals', 'accompaniment']
        elif self.n_stems == 4:
            return ['vocals', 'drums', 'bass', 'other']
        else:  # 5 stems
            return ['vocals', 'drums', 'bass', 'piano', 'other']


# =============================================================================
# COMMAND-LINE SEPARATOR (Generic)
# =============================================================================

class CLISeparator(SourceSeparator):
    """
    Generic adapter for command-line separation tools.

    Use this for tools that work via command-line interface,
    writing stems to separate files.

    Usage:
        separator = CLISeparator(
            command_template='demucs {input} -o {output}',
            stem_patterns={'drums': '*drums.wav', 'bass': '*bass.wav'}
        )
    """

    def __init__(
        self,
        command_template: str,
        stem_patterns: Dict[str, str],
        input_format: str = 'wav'
    ):
        """
        Initialize CLI separator

        Args:
            command_template: Command template with {input} and {output} placeholders
            stem_patterns: Mapping from stem types to file patterns
            input_format: Format for input audio file
        """
        self.command_template = command_template
        self.stem_patterns = stem_patterns
        self.input_format = input_format

    def separate(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Separate audio using CLI tool

        Args:
            audio: Audio waveform
            sr: Sample rate

        Returns:
            Dictionary with separated stems
        """
        import soundfile as sf
        import glob

        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Write input audio
            input_path = tmpdir_path / f'input.{self.input_format}'
            sf.write(str(input_path), audio, sr)

            # Run separation command
            command = self.command_template.format(
                input=str(input_path),
                output=str(tmpdir_path)
            )

            logger.info(f"Running: {command}")
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logger.error(f"Separation failed: {result.stderr}")
                raise RuntimeError(f"CLI separation failed: {result.stderr}")

            # Read stems
            stems = {}
            for stem_type, pattern in self.stem_patterns.items():
                matches = glob.glob(str(tmpdir_path / pattern))
                if matches:
                    stem_audio, _ = sf.read(matches[0])
                    # Convert to mono if stereo
                    if stem_audio.ndim == 2:
                        stem_audio = np.mean(stem_audio, axis=1)
                    stems[stem_type] = stem_audio
                else:
                    logger.warning(f"No file found for stem '{stem_type}' with pattern '{pattern}'")

            logger.info(f"Loaded {len(stems)} stems from CLI output")

            return stems

    @property
    def available_stems(self) -> List[str]:
        return list(self.stem_patterns.keys())


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_separator(
    method: str = 'mock',
    **kwargs
) -> SourceSeparator:
    """
    Factory function to create a source separator

    Args:
        method: Separation method ('mock', 'demucs', 'spleeter', 'cli')
        **kwargs: Method-specific arguments

    Returns:
        SourceSeparator instance

    Examples:
        # Mock (for testing)
        separator = create_separator('mock')

        # Demucs
        separator = create_separator('demucs', model='htdemucs', device='cuda')

        # Spleeter
        separator = create_separator('spleeter', stems=4)

        # Custom CLI tool
        separator = create_separator(
            'cli',
            command_template='my-tool {input} -o {output}',
            stem_patterns={'drums': '*drums.wav', 'bass': '*bass.wav'}
        )
    """
    if method == 'mock':
        return MockSeparator()

    elif method == 'demucs':
        return DemucsSeparator(**kwargs)

    elif method == 'spleeter':
        return SpleeterSeparator(**kwargs)

    elif method == 'cli':
        return CLISeparator(**kwargs)

    else:
        raise ValueError(
            f"Unknown separation method: {method}. "
            f"Choose from: mock, demucs, spleeter, cli"
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def match_instrument_to_stem(program: int, is_drum: bool = False) -> str:
    """
    Match MIDI instrument program to appropriate stem type

    Args:
        program: MIDI program number (0-127)
        is_drum: Whether this is a drum track

    Returns:
        Stem type identifier
    """
    if is_drum:
        return StemType.DRUMS

    return StemType.get_stem_for_program(program)


def get_stem_audio(
    stems: Dict[str, np.ndarray],
    stem_type: str,
    fallback_to_other: bool = True
) -> Optional[np.ndarray]:
    """
    Get audio for a specific stem type with fallback

    Args:
        stems: Dictionary of separated stems
        stem_type: Desired stem type
        fallback_to_other: If True, return 'other' stem if requested stem not available

    Returns:
        Audio array or None
    """
    if stem_type in stems:
        return stems[stem_type]

    if fallback_to_other and StemType.OTHER in stems:
        logger.debug(f"Stem '{stem_type}' not available, using '{StemType.OTHER}'")
        return stems[StemType.OTHER]

    return None


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("SOURCE SEPARATION INTERFACE - DEMO")
    print("=" * 70)

    # Generate synthetic audio
    sr = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

    print("\n1. Mock Separator (for testing)")
    separator = create_separator('mock')
    stems = separator.separate(audio, sr)
    print(f"   Stems: {list(stems.keys())}")
    print(f"   Each stem duration: {len(stems['bass']) / sr:.2f}s")

    print("\n2. MIDI Program to Stem Mapping")
    test_programs = [0, 24, 33, 56, 52, 100]
    for program in test_programs:
        stem = match_instrument_to_stem(program)
        print(f"   Program {program:3d} → Stem: {stem}")

    print("\n3. Integration Example")
    print("""
    # In your code:
    from laplace_mrmt3.source_separation import create_separator

    # Create separator (choose your method)
    separator = create_separator('demucs', model='htdemucs', device='cuda')

    # Separate audio
    stems = separator.separate(audio, sr=16000)

    # Use in ML refinement pipeline
    # (See updated ml_refinement.py)
    """)

    print("\n" + "=" * 70)
