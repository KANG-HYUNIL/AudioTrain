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

@TRANSCRIBER_REGISTRY.register("yourmt3")
def _reg_yourmt3(wav: torch.Tensor, sr: int, instrument: str) -> MidiTrack:
    """Registry entry for YourMT3 backend (Cascading mode).
    
    Runs the full E2E model on the stem and filters for the requested instrument.
    """
    # Run full transcription
    # We don't pass a specific model path here, relying on defaults or env vars for now
    # In a real scenario, we might want to inject config here.
    bundle = _transcribe_yourmt3(wav, sr, model_path=None, vocab_path=None)
    
    # Filter for the requested instrument
    # 1. Try exact match
    if instrument in bundle.tracks:
        return bundle.tracks[instrument]
    
    # 2. Try case-insensitive match
    for name, track in bundle.tracks.items():
        if name.lower() == instrument.lower():
            track.instrument = instrument # Normalize name
            return track
            
    # 3. Heuristic: If we asked for 'drums', look for 'Drums' or program 128
    #    If we asked for 'piano', look for 'Piano' or program 0
    #    For now, if we can't find it, return the track with the most notes 
    #    (assuming the stem contains mostly that instrument)
    best_track = None
    max_notes = -1
    
    for name, track in bundle.tracks.items():
        # Simple heuristic mapping
        if instrument == "drums" and ("drum" in name.lower()):
            return track
        if instrument == "piano" and ("piano" in name.lower()):
            return track
            
        if len(track.notes) > max_notes:
            max_notes = len(track.notes)
            best_track = track
            
    if best_track:
        log.info(f"[YourMT3] Requested '{instrument}' but found {[t for t in bundle.tracks]}. Returning '{best_track.instrument}' as best guess.")
        best_track.instrument = instrument
        return best_track
        
    return MidiTrack(instrument=instrument, notes=[])


@TRANSCRIBER_REGISTRY.register("mt3")
def _reg_mt3(wav: torch.Tensor, sr: int, instrument: str) -> MidiTrack:
    """Registry entry for MT3 backend (Cascading mode)."""
    bundle = _transcribe_mt3(wav, sr, model_path=None)
    
    # Similar filtering logic as YourMT3
    if instrument in bundle.tracks:
        return bundle.tracks[instrument]
        
    for name, track in bundle.tracks.items():
        if name.lower() == instrument.lower():
            track.instrument = instrument
            return track
            
    # Fallback
    best_track = None
    max_notes = -1
    for name, track in bundle.tracks.items():
        if len(track.notes) > max_notes:
            max_notes = len(track.notes)
            best_track = track
            
    if best_track:
        best_track.instrument = instrument
        return best_track
        
    return MidiTrack(instrument=instrument, notes=[])


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


# Global cache for models to avoid reloading per stem in cascading mode
_MT3_MODEL_CACHE = None
_YOURMT3_MODEL_CACHE = None


def _transcribe_mt3(waveform: torch.Tensor, sr: int, model_path: Optional[str]) -> MidiBundle:
    """Transcribe using MT3 (Music Transformer) via mt3-pytorch submodule.
    
    Requires 'external/mt3-pytorch' submodule and dependencies (ddsp, t5, note-seq).
    """
    global _MT3_MODEL_CACHE
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
    
    try:
        # Save waveform
        wav = _to_mono_tensor(waveform)
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
            sf.write(tmp_wav_path, wav.numpy(), target_sr)

        # 4. Initialize Model (Cached)
        if _MT3_MODEL_CACHE is None:
            if model_path is None:
                # Default to a 'checkpoints/mt3' dir if not specified
                model_path = str(project_root / "checkpoints" / "mt3")
                log.warning(f"[MT3] No model_path provided. Trying default: {model_path}")
            
            if not os.path.exists(model_path):
                 log.error(f"[MT3] Model checkpoint not found at {model_path}.")
                 return MidiBundle()

            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info(f"[MT3] Loading model from {model_path} on {device}...")
            
            _MT3_MODEL_CACHE = InferenceHandler(
                weight_path=model_path,
                device=torch.device(device)
            )
        
        inference = _MT3_MODEL_CACHE

        # 5. Run Inference
        log.info(f"[MT3] Running inference...")
        inference.inference(
            audio_path=tmp_wav_path,
            outpath=tmp_mid_path,
            valid_programs=None, # Generate all instruments
            num_beams=1
        )
        
        # 6. Parse Result
        # Check for .mid extension appending behavior
        output_file = tmp_mid_path
        if not os.path.exists(output_file) and os.path.exists(output_file + ".mid"):
            output_file += ".mid"
            
        if not os.path.exists(output_file):
             log.error("[MT3] Output MIDI file was not created.")
             return MidiBundle()

        pm = pretty_midi.PrettyMIDI(output_file)
        bundle = MidiBundle()
        
        for instr in pm.instruments:
            # Determine instrument name
            name = instr.name.strip()
            if not name:
                try:
                    name = pretty_midi.program_to_instrument_name(instr.program)
                except:
                    name = f"program_{instr.program}"
            
            # Convert notes
            notes = []
            for note in instr.notes:
                notes.append((note.start, note.end, note.pitch, note.velocity))
            
            # Handle duplicate names
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
    """Transcribe using YourMT3 backend.
    
    Requires 'external/YourMT3' and its dependencies.
    """
    global _YOURMT3_MODEL_CACHE
    t0 = time.time()
    log.info("[YourMT3] Starting transcription...")
    
    # 1. Setup paths
    project_root = Path(__file__).parent.parent
    yourmt3_dir = project_root / "external" / "YourMT3"
    
    if not yourmt3_dir.exists():
        log.error(f"[YourMT3] Directory not found at {yourmt3_dir}.")
        return MidiBundle()

    # Add to sys.path
    if str(yourmt3_dir) not in sys.path:
        sys.path.append(str(yourmt3_dir))
        # Also add amt/src as done in YourMT3/app.py
        amt_src = yourmt3_dir / "amt" / "src"
        if amt_src.exists() and str(amt_src) not in sys.path:
            sys.path.append(str(amt_src))

    try:
        import model_helper
        import pretty_midi
    except ImportError as e:
        log.error(f"[YourMT3] Failed to import modules: {e}. Check requirements.")
        return MidiBundle()

    # 2. Prepare Audio (Save to temp WAV)
    # YourMT3 requires a file path for input
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tmp_wav_path = tmp_wav.name
        
    try:
        # Save waveform
        wav = _to_mono_tensor(waveform)
        # YourMT3 handles resampling, but saving at a standard rate is good.
        try:
            import torchaudio
            if wav.dim() == 1:
                wav_save = wav.unsqueeze(0)
            else:
                wav_save = wav
            torchaudio.save(tmp_wav_path, wav_save, sr)
        except ImportError:
            import soundfile as sf
            sf.write(tmp_wav_path, wav.numpy(), sr)

        # 3. Initialize Model (Cached)
        if _YOURMT3_MODEL_CACHE is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info(f"[YourMT3] Loading model on {device}...")
            
            # Construct args for load_model_checkpoint
            # We use the default "YPTF.MoE+Multi (noPS)" configuration from app.py
            # unless model_path is provided.
            
            # Default checkpoint name in YourMT3 repo
            default_ckpt = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
            
            if model_path:
                checkpoint = model_path
            else:
                # Try to find the default checkpoint in checkpoints/yourmt3 or external/YourMT3
                possible_paths = [
                    project_root / "checkpoints" / "yourmt3" / default_ckpt,
                    yourmt3_dir / default_ckpt
                ]
                checkpoint = None
                for p in possible_paths:
                    if p.exists():
                        checkpoint = str(p)
                        break
                
                if checkpoint is None:
                    log.warning(f"[YourMT3] Checkpoint {default_ckpt} not found. Attempting to download or fail...")
                    # For now, let's assume the user has it or let model_helper fail
                    checkpoint = default_ckpt 

            # Args based on "YPTF.MoE+Multi (noPS)"
            # Note: These args must match the checkpoint architecture!
            args = [
                checkpoint, 
                '-p', '2024', 
                '-tk', 'mc13_full_plus_256', 
                '-dec', 'multi-t5',
                '-nl', '26', 
                '-enc', 'perceiver-tf', 
                '-sqr', '1', 
                '-ff', 'moe',
                '-wf', '4', 
                '-nmoe', '8', 
                '-kmoe', '2', 
                '-act', 'silu', 
                '-epe', 'rope',
                '-rp', '1', 
                '-ac', 'spec', 
                '-hop', '300', 
                '-atc', '1', 
                '-pr', '16' if device == 'cuda' else '32'
            ]
            
            # Temporarily change CWD to YourMT3 dir because it might rely on relative paths for config/vocab
            original_cwd = os.getcwd()
            os.chdir(yourmt3_dir)
            try:
                _YOURMT3_MODEL_CACHE = model_helper.load_model_checkpoint(args=args, device=device)
            finally:
                os.chdir(original_cwd)
        
        model = _YOURMT3_MODEL_CACHE

        # 4. Run Inference
        log.info(f"[YourMT3] Running inference...")
        
        # Prepare audio_info dict
        audio_info = {
            'filepath': tmp_wav_path,
            'track_name': 'temp_output'
        }
        
        # YourMT3 writes output to ./model_output/ relative to CWD
        # We should control this. model_helper.transcribe writes to './model_output/'
        # We need to handle this side effect.
        
        # Create a temp dir for output to avoid cluttering
        with tempfile.TemporaryDirectory() as temp_out_dir:
            # We need to monkeypatch or change CWD because transcribe hardcodes './model_output/'
            # Actually, looking at model_helper.py:
            # midifile =  os.path.join('./model_output/', audio_info['track_name']  + '.mid')
            # It creates ./model_output/ if not exists inside write_model_output_as_midi
            
            original_cwd = os.getcwd()
            os.chdir(temp_out_dir)
            try:
                # Run transcription
                # Note: model_helper.transcribe prints a lot and uses Timer
                midi_path = model_helper.transcribe(model, audio_info)
            finally:
                os.chdir(original_cwd)
                
            # The midi_path returned is relative to temp_out_dir (e.g. "./model_output/temp_output.mid")
            full_midi_path = os.path.join(temp_out_dir, midi_path)
            
            if not os.path.exists(full_midi_path):
                log.error(f"[YourMT3] Output MIDI not found at {full_midi_path}")
                return MidiBundle()
                
            # 5. Parse Result
            pm = pretty_midi.PrettyMIDI(full_midi_path)
            bundle = MidiBundle()
            
            for instr in pm.instruments:
                name = instr.name.strip()
                if not name:
                    try:
                        name = pretty_midi.program_to_instrument_name(instr.program)
                    except:
                        name = f"program_{instr.program}"
                
                notes = []
                for note in instr.notes:
                    notes.append((note.start, note.end, note.pitch, note.velocity))
                
                base_name = name
                idx = 1
                while name in bundle.tracks:
                    name = f"{base_name}_{idx}"
                    idx += 1
                
                bundle.tracks[name] = MidiTrack(instrument=name, notes=notes)

            duration = time.time() - t0
            total_notes = sum(len(t.notes) for t in bundle.tracks.values())
            log.info(f"[YourMT3] Finished in {duration:.2f}s. Found {len(bundle.tracks)} tracks, {total_notes} notes.")
            return bundle

    except Exception as e:
        log.error(f"[YourMT3] Transcription failed: {e}", exc_info=True)
        return MidiBundle()
    finally:
        if os.path.exists(tmp_wav_path):
            try:
                os.remove(tmp_wav_path)
            except OSError:
                pass

def _transcribe_perceiver_tf(waveform: torch.Tensor, sr: int, model_path: Optional[str]) -> MidiBundle:
    """Stub for Perceiver TF transcription."""
    t0 = time.time()
    log.info("[PerceiverTF] Starting transcription (Stub)...")
    
    # TODO: Implement Perceiver inference
    
    bundle = MidiBundle()
    duration = time.time() - t0
    log.info(f"[PerceiverTF] Finished in {duration:.2f}s. (Placeholder: No notes generated)")
    return bundle






