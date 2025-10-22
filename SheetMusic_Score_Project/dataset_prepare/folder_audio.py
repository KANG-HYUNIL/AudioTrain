"""
Folder-based audio classification dataset.

Expected layout:
  <data_dir>/<class_name>/*.wav (or other supported audio files)

This dataset:
- infers classes from subfolders (or uses a provided class list),
- loads audio with torchaudio.load,
- optionally applies a transform callable(waveform, src_sr) -> features,
- returns (features, label_index) or (features, label_index, path) if return_path=True.
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torchaudio
from torch.utils.data import Dataset

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def _list_audio_files(dirpath: Path) -> List[Path]:
    """Recursively list supported audio files under dirpath."""
    return [p for p in dirpath.rglob("*") if p.suffix.lower() in AUDIO_EXTS and p.is_file()]


def scan_items(
    data_dir: Path,
    classes: Optional[Sequence[str]] = None,
    max_files_per_class: Optional[int] = None,
) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    """
    Scan folder structure and build (filepath, label_name) list and class_to_idx mapping.

    Args:
        data_dir: Root folder containing class subfolders.
        classes: Optional fixed list of class names (subfolder names). If None, infer.
        max_files_per_class: Optional cap per class for quick experiments.

    Returns:
        items: List of (filepath, label_name).
        class_to_idx: Mapping from label_name to integer index.
    """
    data_dir = Path(data_dir)
    if classes is None:
        classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    items: List[Tuple[str, str]] = []
    for c in classes:
        cdir = data_dir / c
        if not cdir.exists():
            continue
        files = sorted(_list_audio_files(cdir))
        if max_files_per_class is not None:
            files = files[:max_files_per_class]
        items.extend([(str(p), c) for p in files])

    labels = sorted({lbl for _, lbl in items})
    class_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    return items, class_to_idx


class FolderAudioDataset(Dataset):
    """
    A simple folder-based audio classification dataset.

    It loads waveforms with torchaudio, applies an optional transform, and returns
    (features, label_index) or (features, label_index, path) when return_path=True.
    """

    def __init__(
        self,
        data_dir: Path,
        classes: Optional[Sequence[str]] = None,
        max_files_per_class: Optional[int] = None,
        transform: Optional[Callable] = None,
        return_path: bool = False,
    ):
        """
        Args:
            data_dir: Root directory with class subfolders.
            classes: Optional explicit class list (subfolder names).
            max_files_per_class: Optional per-class cap for speed/memory control.
            transform: Optional callable(waveform, src_sr) -> features tensor.
            return_path: If True, __getitem__ returns (features, label, path).
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.return_path = return_path

        self.items, self.class_to_idx = scan_items(
            self.data_dir, classes=classes, max_files_per_class=max_files_per_class
        )
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        if len(self.items) == 0:
            raise RuntimeError(f"No audio files found under: {self.data_dir}")

        # Basic stats
        per_class: Dict[str, int] = {}
        for _, lbl in self.items:
            per_class[lbl] = per_class.get(lbl, 0) + 1
        print(f"[FolderAudioDataset] samples={len(self.items)} classes={len(self.class_to_idx)} dist={per_class}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, label_name = self.items[index]
        label_idx = self.class_to_idx[label_name]

        # Load waveform (C, T) and sampling rate
        waveform, sr = torchaudio.load(path)

        # Apply transform if provided
        if self.transform is not None:
            try:
                features = self.transform(waveform, src_sr=sr)
            except TypeError:
                # Backward-compatible signatures
                try:
                    features = self.transform(waveform, sr)
                except Exception:
                    features = self.transform(waveform)
        else:
            features = waveform

        if self.return_path:
            return features, label_idx, path
        return features, label_idx
