"""Packaging utilities: merge per-instrument tracks and export to files.

Uses pretty_midi for MIDI export when available; falls back to placeholder.
MusicXML export remains a lightweight placeholder to be replaced by music21.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
from pathlib import Path

from .transcription import MidiBundle, MidiTrack
import importlib
import numpy as np


def _gm_program_for(instrument: str) -> int:
    """Return a General MIDI program index (0-127) for a given instrument label.

    This is a heuristic mapping; adjust as needed.
    """
    name = instrument.lower()
    if "piano" in name:
        return 0  # Acoustic Grand Piano
    if "guitar" in name:
        return 24  # Acoustic Guitar (nylon)
    if "bass" in name:
        return 32  # Acoustic Bass (fingered)
    if any(k in name for k in ["violin", "strings", "cello", "viola"]):
        return 48  # Strings Ensemble 1
    if "organ" in name:
        return 16  # Drawbar Organ
    if "synth" in name:
        return 80  # Synth Lead (square)
    if "voice" in name or "vocal" in name or "choir" in name:
        return 52  # Choir Aahs
    return 0


def _parse_time_signature(sig: Optional[str]) -> Optional[Tuple[int, int]]:
    """Parse a time signature string like '4/4' into (numerator, denominator).

    Returns None if parsing fails.
    """
    if not sig or not isinstance(sig, str) or "/" not in sig:
        return None
    try:
        num_s, den_s = sig.strip().split("/", 1)
        num = int(num_s)
        den = int(den_s)
        if num > 0 and den in (1, 2, 4, 8, 16):
            return num, den
    except Exception:
        return None
    return None


def _scale_velocity(instrument: str, velocity: int) -> int:
    """Apply simple per-instrument velocity scaling policy.

    Heuristic mapping to get more natural dynamics by instrument family.
    """
    v = max(1, min(127, int(velocity)))
    x = v / 127.0
    name = (instrument or "").lower()
    # Default parameters
    gain = 1.0
    gamma = 1.0
    if "piano" in name:
        gain, gamma = 1.0, 0.95   # keep wide dynamics
    elif "guitar" in name:
        gain, gamma = 0.85, 1.0   # slightly compressed
    elif "bass" in name:
        gain, gamma = 0.75, 1.05  # lower overall, tiny curve
    elif any(k in name for k in ["violin", "strings", "cello", "viola"]):
        gain, gamma = 0.9, 0.95
    elif any(k in name for k in ["voice", "vocal", "choir"]):
        gain, gamma = 0.85, 1.0
    elif "drum" in name:
        gain, gamma = 1.0, 1.0    # leave as-is
    y = (x ** gamma) * gain
    y = max(0.01, min(1.0, y))
    return int(max(1, min(127, round(y * 127))))


def merge_tracks(tracks: Dict[str, MidiTrack], tempo_bpm: Optional[float] = None, metadata: Optional[dict] = None, time_signature: Optional[str] = None) -> MidiBundle:
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
    # Attach optional metadata for downstream exporters
    setattr(bundle, "tempo_bpm", tempo_bpm)
    setattr(bundle, "time_signature", time_signature)
    if metadata is not None:
        setattr(bundle, "metadata", dict(metadata))
    return bundle


def export_midi(bundle: MidiBundle, output_path: str | Path) -> str:
    """Export a MidiBundle to a MIDI file using pretty_midi when available.

    Falls back to creating a placeholder file if pretty_midi is not installed.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        pm = importlib.import_module("pretty_midi")
    except Exception:
        out.write_bytes(b"Placeholder MIDI content. Install pretty_midi for real export.")
        return str(out)

    midi = pm.PrettyMIDI()
    # If tempo info exists in bundle, insert a global tempo change
    # Note: Our MidiBundle currently stores tempo via merge_tracks argument; we can attach via a meta track concept later.
    # Here we just set an initial tempo if provided by caller via merge_tracks.
    try:
        tempo_bpm = getattr(bundle, "tempo_bpm", None)
    except Exception:
        tempo_bpm = None
    # Note: PrettyMIDI automatically infers tempo if not set; explicit tempo event insertion is optional and deferred.
    # Time signature change at t=0 if provided (some DAWs read it)
    ts = _parse_time_signature(getattr(bundle, "time_signature", None))
    if ts is not None and hasattr(pm, "TimeSignature"):
        try:
            midi.time_signature_changes.append(pm.TimeSignature(ts[0], ts[1], 0.0))
        except Exception:
            pass

    for name, track in bundle.tracks.items():
        is_drum = "drum" in name.lower()
        program = 0 if is_drum else _gm_program_for(name)
        inst = pm.Instrument(program=program, is_drum=is_drum, name=name)
        for onset, offset, pitch, velocity in track.notes:
            onset_s = max(0.0, float(onset))
            offset_s = max(onset_s + 1e-3, float(offset))
            vel_scaled = _scale_velocity(name, int(velocity))
            note = pm.Note(velocity=int(vel_scaled), pitch=int(pitch), start=onset_s, end=offset_s)
            inst.notes.append(note)
        midi.instruments.append(inst)
    midi.write(str(out))
    return str(out)


def export_musicxml(bundle: MidiBundle, output_path: str | Path) -> str:
    """Export to MusicXML using music21 with a simple multi-part score.

    If music21 is not available, writes a placeholder file. This implementation
    creates one Part per track and inserts a global metronome mark if a tempo is
    provided via bundle. Note durations are expressed in quarterLength using the
    default 4/4 assumptions.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        m21 = importlib.import_module("music21")
    except Exception:
        out.write_text("<!-- Placeholder MusicXML. Install music21 for real export. -->\n")
        return str(out)

    score = m21.stream.Score()
    # Tempo
    tempo_bpm = getattr(bundle, "tempo_bpm", None)
    if tempo_bpm is not None:
        mm = m21.tempo.MetronomeMark(number=float(tempo_bpm))
        score.insert(0, mm)
    # Time signature
    ts_tuple = _parse_time_signature(getattr(bundle, "time_signature", None))
    if ts_tuple is None:
        ts_tuple = (4, 4)

    # Build parts
    for name, track in bundle.tracks.items():
        part = m21.stream.Part(id=name)
        instr = m21.instrument.fromString(name) if hasattr(m21.instrument, 'fromString') else m21.instrument.Instrument()
        instr.partName = name
        part.insert(0, instr)

        num, den = ts_tuple
        measure_len = float(num) * (4.0 / float(den))  # in quarterLength units
        cur_len = 0.0
        m_number = 1
        meas = m21.stream.Measure(number=m_number)
        meas.append(m21.meter.TimeSignature(f"{num}/{den}"))

        for onset, offset, pitch, velocity in track.notes:
            dur_sec = max(1e-3, float(offset) - float(onset))
            ql = max(1e-3, dur_sec * (float(tempo_bpm) / 60.0)) if tempo_bpm else dur_sec
            n = m21.note.Note(int(pitch))
            n.quarterLength = ql
            n.volume.velocity = _scale_velocity(name, int(velocity))

            if cur_len + ql > measure_len and cur_len > 0:
                # Close current measure and start a new one
                part.append(meas)
                m_number += 1
                meas = m21.stream.Measure(number=m_number)
                cur_len = 0.0
            meas.append(n)
            cur_len += ql

        # Flush last measure
        if len(list(meas.notes)) > 0:
            part.append(meas)
        score.append(part)

    score.write('musicxml', fp=str(out))
    return str(out)
