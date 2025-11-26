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
import logging
import time
import tempfile
import os
import sys

from .utils import Registry

log = logging.getLogger(__name__)

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
    t0 = time.time()
    log.debug(f"[BasicPitch] Starting transcription for {instrument}")

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
        log.debug(f"Running Basic Pitch prediction for {instrument}...")
        # basic_pitch.inference.predict expects (audio_array, model_or_model_path, ...)
        # It assumes 22050Hz input when passing an array. We do not pass sr as the second arg.
        model_output, midi_data, note_events = bp_predict(audio_np)
    except Exception as e:
        log.error(f"Basic Pitch prediction failed for {instrument}: {e}")
        # Fallback to empty return if prediction crashes
        return MidiTrack(instrument=instrument, notes=[])

    notes = _bp_note_events_to_list(note_events)
    
    duration = time.time() - t0
    log.debug(f"[BasicPitch] Finished for {instrument} in {duration:.2f}s. Found {len(notes)} notes.")
    return MidiTrack(instrument=instrument, notes=notes)


def transcribe_onsets_frames(waveform: torch.Tensor, sr: int) -> MidiTrack:
    """Transcribe using an Onsets-and-Frames implementation (piano-focused).

    This function uses the 'piano_transcription_inference' package if available.
    It resamples the audio to the package's expected sample rate and converts
    predicted note events to our normalized format.
    """
    t0 = time.time()
    log.debug(f"[OnsetsFrames] Starting transcription")
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
    log.debug("Running Onsets-and-Frames transcription...")
    out = model.transcribe(wav.numpy(), output_midi_path=None)  # type: ignore[arg-type]
    # Try common keys for note events
    note_events = out.get("note_events") or out.get("est_note_events") or []
    notes = _bp_note_events_to_list(note_events)
    
    duration = time.time() - t0
    log.debug(f"[OnsetsFrames] Finished in {duration:.2f}s. Found {len(notes)} notes.")
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
    t0 = time.time()
    log.debug(f"[CREPE] Starting transcription for {instrument}")
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

    log.debug(f"Running CREPE prediction for {instrument}...")
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
    
    duration = time.time() - t0
    log.debug(f"[CREPE] Finished for {instrument} in {duration:.2f}s. Found {len(notes)} notes.")
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


@TRANSCRIBER_REGISTRY.register("omnizart")
def _reg_omnizart(wav: torch.Tensor, sr: int, instrument: str) -> MidiTrack:
    """Registry entry for Omnizart backend."""
    return _transcribe_omnizart(wav, sr, instrument=instrument)


def _transcribe_omnizart(waveform: torch.Tensor, sr: int, instrument: str) -> MidiTrack:
    """Transcribe using Omnizart (General, Vocal, or Drum models).
    
    Requires 'omnizart' package and checkpoints (omnizart download-checkpoints).
    """
    t0 = time.time()
    log.debug(f"[Omnizart] Starting transcription for {instrument}")
    
    # Lazy import
    try:
        from omnizart.music import app as music_app
        from omnizart.drum import app as drum_app
        from omnizart.vocal import app as vocal_app
    except ImportError as e:
        raise RuntimeError(
            "omnizart is not installed. Install with: pip install omnizart"
        ) from e

    # Map instrument to Omnizart module
    # 'music' is for polyphonic instruments (piano, guitar, etc.)
    # 'vocal' is for monophonic voice
    # 'drum' is for percussion
    mode = "music"
    if instrument == "drums":
        mode = "drum"
    elif instrument == "vocals":
        mode = "vocal"
    
    # Prepare temp file because Omnizart expects file path
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Save waveform to temp file
        wav = _to_mono_tensor(waveform)
        
        # Use torchaudio or soundfile to save
        try:
            import torchaudio
            if wav.dim() == 1:
                wav_save = wav.unsqueeze(0)
            else:
                wav_save = wav
            torchaudio.save(tmp_path, wav_save, sr)
        except ImportError:
            import soundfile as sf
            sf.write(tmp_path, wav.numpy(), sr)

        # Run Inference
        log.debug(f"Running Omnizart ({mode}) prediction...")
        midi_data = None
        
        if mode == "music":
            app = music_app.MusicTranscription()
            midi_data = app.transcribe(tmp_path) 
        elif mode == "vocal":
            app = vocal_app.VocalTranscription()
            midi_data = app.transcribe(tmp_path)
        elif mode == "drum":
            app = drum_app.DrumTranscription()
            midi_data = app.transcribe(tmp_path)
            
    except Exception as e:
        log.error(f"[Omnizart] Failed for {instrument}: {e}")
        return MidiTrack(instrument=instrument, notes=[])
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    # Parse pretty_midi object to notes list
    notes = []
    if midi_data:
        # Omnizart returns a pretty_midi.PrettyMIDI object
        for instr_pm in midi_data.instruments:
            for note in instr_pm.notes:
                notes.append((note.start, note.end, note.pitch, note.velocity))
    
    duration = time.time() - t0
    log.debug(f"[Omnizart] Finished for {instrument} in {duration:.2f}s. Found {len(notes)} notes.")
    return MidiTrack(instrument=instrument, notes=notes)


# -----------------------------------------------------------------------------
# End-to-End (Mix) Transcription
# -----------------------------------------------------------------------------

def transcribe_mix_to_bundle(
    waveform: torch.Tensor, 
    sr: int, 
    backend: str = "mt3",
    model_path: Optional[str] = None,
    vocab_path: Optional[str] = None
) -> MidiBundle:
    """Transcribe a mix directly to a MidiBundle using an End-to-End model.
    
    Args:
        waveform: Audio tensor (1, T) or (C, T).
        sr: Sample rate.
        backend: Name of the backend ('mt3', 'yourmt3', 'perceiver_tf').
        model_path: Path to model checkpoint (optional).
        vocab_path: Path to vocabulary file (optional, for YourMT3).
        
    Returns:
        MidiBundle containing transcribed tracks.
    """
    backend = backend.lower()
    log.info(f"Transcribing mix with backend: {backend}")
    
    # Ensure mono
    wav = _to_mono_tensor(waveform)
    
    if backend == "mt3":
        return _transcribe_mt3(wav, sr, model_path)
    elif backend == "yourmt3":
        return _transcribe_yourmt3(wav, sr, model_path, vocab_path)
    elif backend == "perceiver_tf":
        return _transcribe_perceiver_tf(wav, sr, model_path)
    else:
        log.warning(f"Unknown E2E backend: {backend}. Returning empty bundle.")
        return MidiBundle()


def _transcribe_mt3(waveform: torch.Tensor, sr: int, model_path: Optional[str]) -> MidiBundle:
    """Transcribe using MT3 (Music Transformer) via mt3-pytorch submodule.
    
    Requires 'external/mt3-pytorch' submodule and dependencies (ddsp, t5, note-seq).
    """
    t0 = time.time()
    log.info("[MT3] Starting transcription...")

    # 1. Setup paths
    # Assuming 'external/mt3-pytorch' is at the project root relative to this file
    # audio2score/transcription.py -> ../external/mt3-pytorch
    project_root = Path(__file__).parent.parent
    mt3_dir = project_root / "external" / "mt3-pytorch"
    
    if not mt3_dir.exists():
        log.error(f"[MT3] Submodule not found at {mt3_dir}. Please run: git submodule update --init --recursive")
        return MidiBundle()

    # Add submodule to sys.path to import 'inference'
    if str(mt3_dir) not in sys.path:
        sys.path.append(str(mt3_dir))

    try:
        from inference import InferenceHandler
        import pretty_midi
    except ImportError as e:
        log.error(f"[MT3] Failed to import mt3-pytorch modules: {e}. Check requirements.")
        return MidiBundle()

    # 2. Prepare Audio (Save to temp WAV)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tmp_wav_path = tmp_wav.name
    
    # 3. Prepare Output Path
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp_mid:
        tmp_mid_path = tmp_mid.name
    # InferenceHandler appends .mid extension automatically if not present, 
    # but let's handle it carefully. The library code does:
    # if outpath is None: ... else: os.makedirs(...); note_seq...to_midi_file(..., outpath)
    # So we should provide the full path.

    try:
        # Save waveform
        wav = _to_mono_tensor(waveform)
        # MT3 expects 16kHz usually, but the InferenceHandler loads with librosa at self.SAMPLE_RATE (16000)
        # So we can save at current SR, librosa will resample inside InferenceHandler.
        # However, saving at 16k is safer/faster.
        target_sr = 16000
        try:
            import torchaudio
            if wav.dim() == 1:
                wav_save = wav.unsqueeze(0)
            else:
                wav_save = wav
            if sr != target_sr:
                wav_save = torchaudio.functional.resample(wav_save, sr, target_sr)
            torchaudio.save(tmp_wav_path, wav_save, target_sr)
        except ImportError:
            import soundfile as sf
            # Resample with librosa if needed, or just save and let mt3 handle it
            # Let's just save and let mt3 (librosa) handle resampling to avoid extra deps here if possible
            sf.write(tmp_wav_path, wav.numpy(), sr)

        # 4. Initialize Model
        if model_path is None:
            # Default to a 'checkpoints/mt3' dir if not specified
            model_path = str(project_root / "checkpoints" / "mt3")
            log.warning(f"[MT3] No model_path provided. Trying default: {model_path}")
        
        if not os.path.exists(model_path):
             log.error(f"[MT3] Model checkpoint not found at {model_path}.")
             return MidiBundle()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"[MT3] Loading model from {model_path} on {device}...")
        
        # InferenceHandler expects the directory containing 'mt3.pth' and 'config.json'
        inference = InferenceHandler(
            weight_path=model_path,
            device=torch.device(device)
        )

        # 5. Run Inference
        log.info(f"[MT3] Running inference...")
        inference.inference(
            audio_path=tmp_wav_path,
            outpath=tmp_mid_path,
            valid_programs=None, # Generate all instruments
            num_beams=1
        )
        
        # 6. Parse Result
        if not os.path.exists(tmp_mid_path):
            # Sometimes it might append .mid?
            if os.path.exists(tmp_mid_path + ".mid"):
                tmp_mid_path += ".mid"
            else:
                log.error("[MT3] Output MIDI file was not created.")
                return MidiBundle()

        pm = pretty_midi.PrettyMIDI(tmp_mid_path)
        bundle = MidiBundle()
        
        for instr in pm.instruments:
            # Determine instrument name
            name = instr.name.strip()
            if not name:
                # Fallback to program name
                try:
                    name = pretty_midi.program_to_instrument_name(instr.program)
                except:
                    name = f"program_{instr.program}"
            
            # Convert notes
            notes = []
            for note in instr.notes:
                notes.append((note.start, note.end, note.pitch, note.velocity))
            
            # Handle duplicate names by appending index
            base_name = name
            idx = 1
            while name in bundle.tracks:
                name = f"{base_name}_{idx}"
                idx += 1
            
            bundle.tracks[name] = MidiTrack(instrument=name, notes=notes)
            
        duration = time.time() - t0
        total_notes = sum(len(t.notes) for t in bundle.tracks.values())
        log.info(f"[MT3] Finished in {duration:.2f}s. Found {len(bundle.tracks)} tracks, {total_notes} notes.")
        return bundle

    except Exception as e:
        log.error(f"[MT3] Transcription failed: {e}", exc_info=True)
        return MidiBundle()
    finally:
        # Cleanup temps
        for p in [tmp_wav_path, tmp_mid_path, tmp_mid_path + ".mid"]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def _transcribe_yourmt3(waveform: torch.Tensor, sr: int, model_path: Optional[str], vocab_path: Optional[str]) -> MidiBundle:
    """Stub for YourMT3 transcription.
    
    YourMT3 is a custom T5-based model for arbitrary instrument transcription.
    """
    t0 = time.time()
    log.info("[YourMT3] Starting transcription (Stub)...")
    
    # TODO: Implement YourMT3 inference
    
    bundle = MidiBundle()
    duration = time.time() - t0
    log.info(f"[YourMT3] Finished in {duration:.2f}s. (Placeholder: No notes generated)")
    return bundle


def _transcribe_perceiver_tf(waveform: torch.Tensor, sr: int, model_path: Optional[str]) -> MidiBundle:
    """Stub for Perceiver TF transcription."""
    t0 = time.time()
    log.info("[PerceiverTF] Starting transcription (Stub)...")
    
    # TODO: Implement Perceiver inference
    
    bundle = MidiBundle()
    duration = time.time() - t0
    log.info(f"[PerceiverTF] Finished in {duration:.2f}s. (Placeholder: No notes generated)")
    return bundle





 
