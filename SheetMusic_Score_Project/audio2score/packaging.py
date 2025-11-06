"""Packaging utilities: merge per-instrument tracks and export to files.

Replace this placeholder with pretty_midi or mido for MIDI, and music21 for MusicXML.
"""
from __future__ import annotations
from typing import Dict, Optional
from pathlib import Path

from .transcription import MidiBundle, MidiTrack


def merge_tracks(tracks: Dict[str, MidiTrack], tempo_bpm: Optional[float] = None, metadata: Optional[dict] = None) -> MidiBundle:
    """Merge individual instrument tracks into a multi-track bundle.

    Args:
        tracks: Mapping instrument -> MidiTrack
        tempo_bpm: Optional tempo to embed later
        metadata: Optional free-form metadata dictionary

    Returns:
        MidiBundle with tracks merged (here, just a container copy).
    """
    bundle = MidiBundle()
    bundle.tracks = dict(tracks)
    return bundle


def export_midi(bundle: MidiBundle, output_path: str | Path) -> str:
    """Export a MidiBundle to a MIDI file (placeholder).

    Args:
        bundle: MidiBundle to export
        output_path: Destination .mid path

    Returns:
        The string path to the created file (placeholder writes nothing).
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Placeholder: touch a file to represent the artifact
    out.write_bytes(b"Placeholder MIDI content. Replace with pretty_midi export.")
    return str(out)


def export_musicxml(bundle: MidiBundle, output_path: str | Path) -> str:
    """Export to MusicXML using music21 (placeholder)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("<!-- Placeholder MusicXML. Replace with music21 export. -->\n")
    return str(out)
