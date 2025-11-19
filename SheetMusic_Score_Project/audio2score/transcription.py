"""Transcription backends: Basic Pitch, Onsets-and-Frames, simple monophonic pitch.

This module currently implements Basic Pitch integration for per-stem transcription.
Transcribers accept a waveform tensor and sampling rate, and return a MidiTrack
containing note events that downstream packaging can export to MIDI/MusicXML.
"""
from __future__ import annotations
from typing import Dict, Optional, List, Tuple, Callable, Union
from pathlib import Path

import torch
import math
import importlib
import numpy as np

from .utils import Registry


class MidiTrack:
    """Lightweight MIDI track placeholder.

    This is a struct-like container until we integrate a real MIDI library.
    """
    def __init__(self, instrument: str, notes: Optional[list] = None) -> None:
        self.instrument = instrument
        self.notes = notes or []  # list of (onset_s, offset_s, pitch, velocity)


class MidiBundle:
    """Container for multiple tracks to be packaged later."""
    def __init__(self) -> None:
        self.tracks: Dict[str, MidiTrack] = {}


def transcribe_basic_pitch(audio_path: str | Path) -> MidiTrack:
    """Deprecated path-based transcriber (kept for compatibility).

    Prefer using transcribe_basic_pitch_tensor(waveform, sr). This function
    currently returns an empty track to discourage path-based usage.
    """
    _ = audio_path
    return MidiTrack(instrument="unknown", notes=[])


def _bp_note_events_to_list(note_events: List[Dict]) -> List[Tuple[float, float, int, int]]:
    """Convert Basic Pitch note events dicts into (onset, offset, pitch, velocity).

    Handles small variations in key naming across versions.
    """
    notes: List[Tuple[float, float, int, int]] = []
    for ev in note_events:
        onset = float(ev.get("onset_time", ev.get("start_time", 0.0)))
        offset = float(ev.get("offset_time", ev.get("end_time", onset)))
        # Accept a variety of pitch key names across libraries
        pitch = int(
            ev.get(
                "midi_note",
                ev.get("note_number", ev.get("pitch_midi", ev.get("pitch", 60))),
            )
        )
        amp = float(ev.get("amplitude", ev.get("velocity", 0.8)))
        vel = max(1, min(127, int(round(amp * 127))))
        if offset <= onset:
            offset = onset + 1e-3  # enforce positive duration
        notes.append((onset, offset, pitch, vel))
    return notes


# -----------------------------------------------------------------------------
# Registry and helpers
# -----------------------------------------------------------------------------
# We register backends by name so YAML can select them per instrument.
TRANSCRIBER_REGISTRY = Registry("transcriber")


def _to_mono_tensor(audio: Union[torch.Tensor, np.ndarray, List[float], List[int]]) -> torch.Tensor:
    """Convert various audio array types to a mono 1-D torch.float32 CPU tensor.

    Accepts torch.Tensor, numpy arrays, or Python lists. If multi-channel,
    averages across channel dimension.
    """
    if isinstance(audio, torch.Tensor):
        wav = audio.detach().cpu().float()
        # Expect shapes (C, T) or (T,)
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if wav.dim() == 2:
            wav = wav.squeeze(0)
        return wav.contiguous()
    # numpy or list
    if isinstance(audio, np.ndarray):
        arr = audio.astype(np.float32, copy=False)
    else:
        arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] > 1:
        arr = arr.mean(axis=0, keepdims=False)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    return torch.from_numpy(arr).contiguous()


def transcribe_basic_pitch_tensor(waveform: torch.Tensor, sr: int, instrument: str = "unknown") -> MidiTrack:
    """Transcribe a stem waveform using Basic Pitch.

    Args:
        waveform: (1, T) or (C, T) float32/float tensor in [-1, 1].
        sr: Sampling rate in Hz.
        instrument: Optional instrument label for the resulting MidiTrack.

    Returns:
        MidiTrack with note events derived from Basic Pitch predictions.
    """
    # Lazy-import to avoid static import errors when the package isn't installed
    bp_predict = _get_basic_pitch_predict()

    wav = _to_mono_tensor(waveform)
    # Basic Pitch commonly operates at ~22.05 kHz. Resample for stability.
    target_sr = 22050
    if int(sr) != target_sr:
        try:
            ta = importlib.import_module("torchaudio")
            wav = ta.functional.resample(wav.unsqueeze(0), int(sr), target_sr).squeeze(0)
        except Exception:
            lb = importlib.import_module("librosa")
            wav_np = lb.resample(wav.numpy(), orig_sr=int(sr), target_sr=target_sr)
            wav = torch.from_numpy(wav_np.astype(np.float32))
        sr = target_sr
    # Convert to numpy array for Basic Pitch
    audio_np = wav.numpy()

    # Run prediction
    try:
        model_output, midi_data, note_events = bp_predict(audio_np, sr)  # type: ignore[misc]
    except TypeError:
        # Some versions might expect keyword args
        model_output, midi_data, note_events = bp_predict(audio_np, sample_rate=sr)  # type: ignore[misc]

    notes = _bp_note_events_to_list(note_events)
    return MidiTrack(instrument=instrument, notes=notes)


def transcribe_onsets_frames(waveform: torch.Tensor, sr: int) -> MidiTrack:
    """Transcribe using an Onsets-and-Frames implementation (piano-focused).

    This function uses the 'piano_transcription_inference' package if available.
    It resamples the audio to the package's expected sample rate and converts
    predicted note events to our normalized format.
    """
    wav = _to_mono_tensor(waveform)

    try:
        pti = importlib.import_module("piano_transcription_inference")
        PianoTranscription = getattr(pti, "PianoTranscription")
        # The package exposes a 'sample_rate' constant
        pkg_sr = int(getattr(pti, "sample_rate", 16000))
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "piano_transcription_inference is not installed. Install with: pip install piano_transcription_inference"
        ) from e

    # Resample if needed
    if int(sr) != pkg_sr:
        try:
            ta = importlib.import_module("torchaudio")
            wav_rs = ta.functional.resample(wav.unsqueeze(0), int(sr), pkg_sr).squeeze(0)
        except Exception:
            lb = importlib.import_module("librosa")
            wav_rs_np = lb.resample(wav.numpy(), orig_sr=int(sr), target_sr=pkg_sr)
            wav_rs = torch.from_numpy(wav_rs_np.astype(np.float32))
        wav = wav_rs
        sr = pkg_sr

    # Model inference (CPU by default)
    model = PianoTranscription(device="cpu")
    out = model.transcribe(wav.numpy(), output_midi_path=None)  # type: ignore[arg-type]
    # Try common keys for note events
    note_events = out.get("note_events") or out.get("est_note_events") or []
    notes = _bp_note_events_to_list(note_events)
    return MidiTrack(instrument="piano", notes=notes)


def _hz_to_midi(f0_hz: float) -> int:
    """Convert frequency in Hz to nearest MIDI note number."""
    if f0_hz <= 0:
        return 0
    return int(round(69.0 + 12.0 * math.log2(f0_hz / 440.0)))


def _transcribe_crepe_mono_tensor(waveform: torch.Tensor, sr: int, instrument: str = "unknown") -> MidiTrack:
    """Transcribe a monophonic melody using torchcrepe if available.

    - Resamples to 16 kHz if needed using torchaudio or librosa.
    - Segments voiced regions via periodicity threshold and converts to notes.
    """
    wav = _to_mono_tensor(waveform)

    try:
        tc = importlib.import_module("torchcrepe")
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "torchcrepe is not installed. Install with: pip install torchcrepe"
        ) from e

    target_sr = 16000
    if int(sr) != target_sr:
        try:
            ta = importlib.import_module("torchaudio")
            wav = ta.functional.resample(wav.unsqueeze(0), int(sr), target_sr).squeeze(0)
        except Exception:
            lb = importlib.import_module("librosa")
            wav_np = lb.resample(wav.numpy(), orig_sr=int(sr), target_sr=target_sr)
            wav = torch.from_numpy(wav_np.astype(np.float32))
        sr = target_sr

    hop_length = max(1, int(sr // 100))  # ~10 ms

    # Choose decoder if available
    decoder = None
    if hasattr(tc, "decode") and hasattr(tc.decode, "viterbi"):
        decoder = tc.decode.viterbi

    with torch.no_grad():
        f0, pd = tc.predict(
            wav.unsqueeze(0),
            sr,
            hop_length=hop_length,
            fmin=50.0,
            fmax=2000.0,
            model="full",
            decoder=decoder,
            batch_size=1024,
            device="cpu",
            return_periodicity=True,
        )
    f0 = f0[0].cpu().float()
    pd = pd[0].cpu().float()

    thr = 0.6
    voiced = pd >= thr

    notes: List[Tuple[float, float, int, int]] = []
    if voiced.any():
        dif = torch.diff(voiced.to(torch.int8), prepend=torch.tensor([0], dtype=torch.int8))
        on_idx = (dif == 1).nonzero(as_tuple=False).flatten()
        off_idx = (dif == -1).nonzero(as_tuple=False).flatten()
        if voiced[0]:
            on_idx = torch.cat([torch.tensor([0]), on_idx])
        if voiced[-1]:
            off_idx = torch.cat([off_idx, torch.tensor([voiced.numel()])])
        for s, e in zip(on_idx.tolist(), off_idx.tolist()):
            if e <= s:
                continue
            seg_f0 = f0[s:e]
            seg_pd = pd[s:e]
            onset = s * hop_length / float(sr)
            offset = e * hop_length / float(sr)
            f0_med = float(torch.median(seg_f0[seg_f0 > 0]) if (seg_f0 > 0).any() else 0.0)
            if f0_med <= 0:
                continue
            pitch = _hz_to_midi(f0_med)
            vel = int(max(1, min(127, round(float(seg_pd.mean()) * 127))))
            if offset <= onset:
                offset = onset + 1e-3
            notes.append((onset, offset, pitch, vel))

    return MidiTrack(instrument=instrument, notes=notes)



def pick_transcriber(instrument: str, default: str = "basic_pitch"):
    """Choose a transcriber callable based on backend name and instrument label.

    The returned callable accepts (waveform: Tensor, sr: int) and returns a
    MidiTrack. The instrument label is bound via closure for convenience.
    """
    name = (default or "basic_pitch").lower()
    try:
        backend = TRANSCRIBER_REGISTRY.get(name)
    except KeyError:
        backend = None

    if backend is None:
        # Fallback placeholder returning empty track
        def _of(wav: torch.Tensor, sr: int) -> MidiTrack:
            _ = (wav, sr)
            return MidiTrack(instrument=instrument, notes=[])
        return _of

    def _fn(wav: torch.Tensor, sr: int) -> MidiTrack:
        return backend(wav, sr, instrument)

    return _fn


def _get_basic_pitch_predict() -> Callable:
    """Return basic_pitch.inference.predict function if available.

    Uses importlib to avoid static import errors in environments where
    basic-pitch isn't installed yet. Raises a clear RuntimeError otherwise.
    """
    try:
        mod = importlib.import_module("basic_pitch.inference")
        predict = getattr(mod, "predict", None)
        if predict is None:
            raise AttributeError("'basic_pitch.inference' has no attribute 'predict'")
        return predict
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "basic-pitch is not installed. Install with: pip install basic-pitch pretty_midi"
        ) from e


# -----------------------------------------------------------------------------
# Registered backends
# -----------------------------------------------------------------------------
@TRANSCRIBER_REGISTRY.register("basic_pitch")
def _reg_basic_pitch(wav: torch.Tensor, sr: int, instrument: str) -> MidiTrack:
    """Registry entry for the Basic Pitch backend."""
    return transcribe_basic_pitch_tensor(wav, sr, instrument=instrument)


@TRANSCRIBER_REGISTRY.register("onsets_frames")
def _reg_onsets_frames(wav: torch.Tensor, sr: int, instrument: str) -> MidiTrack:
    """Registry entry for Onsets-and-Frames backend (piano-focused)."""
    # Instrument label can be overridden by caller; events are returned regardless.
    track = transcribe_onsets_frames(wav, sr)
    track.instrument = instrument or track.instrument
    return track

@TRANSCRIBER_REGISTRY.register("crepe_mono")
def _reg_crepe_mono(wav: torch.Tensor, sr: int, instrument: str) -> MidiTrack:
    """Registry entry for torchcrepe monophonic backend."""
    return _transcribe_crepe_mono_tensor(wav, sr, instrument=instrument)





 
