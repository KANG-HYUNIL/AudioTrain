"""
torchaudio 기반 Log-Mel 변환 파이프라인 스텁.
TODO: implement waveform -> log-mel tensor transformation with normalization.
"""

from typing import Optional

import torch


class LogMelSpectrogram(torch.nn.Module):
    def __init__(self, sample_rate: int = 16000, n_mels: int = 128, n_fft: int = 1024, hop_length: int = 256,
                 f_min: float = 20.0, f_max: Optional[float] = 8000.0, log_mel: bool = True, normalize: bool = True):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.log_mel = log_mel
        self.normalize = normalize
        # TODO: create torchaudio.transforms.MelSpectrogram and AmplitudeToDB

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Input: (B, C, T) or (C, T). Output: (B, n_mels, time)."""
        # TODO: implement actual transform with torchaudio
        return waveform  # placeholder
