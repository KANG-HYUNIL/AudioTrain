"""
Log-Mel 변환 파이프라인

입력 waveform(C, T) → mono → (선택) resample → MelSpectrogram → (선택) dB 스케일 → (선택) 정규화

출력 텐서 형태 계약
- 각 샘플 당 (C=1, F=n_mels, T=frames) 3D 텐서로 반환합니다. (배치 차원 없이)
    DataLoader/collate 단계에서 (B, C, F, T)로 스택됩니다.
"""

from typing import Optional

import torch
import torchaudio


class LogMelSpectrogram(torch.nn.Module):
    def __init__(self, sample_rate: int = 16000, n_mels: int = 128, n_fft: int = 1024, hop_length: int = 256,
                 f_min: float = 20.0, f_max: Optional[float] = 8000.0, log_mel: bool = True, normalize: bool = True):
        
        """
        Init the LogMelSpectrogram transform.
        Args:
            sample_rate (int): Sample rate of the input waveform.
            n_mels (int): Number of Mel bands to generate.
            n_fft (int): Size of FFT.
            hop_length (int): Hop length for STFT.
            f_min (float): Minimum frequency for Mel scale.
            f_max (Optional[float]): Maximum frequency for Mel scale. If None, use sample_rate / 2.
            log_mel (bool): Whether to apply logarithmic scaling to Mel spectrogram.
            normalize (bool): Whether to normalize the Mel spectrogram.

        """
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.log_mel = log_mel
        self.normalize = normalize
       
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            f_min=self.f_min,
            f_max=self.f_max,
            power=2.0,
            center=True,
            norm='slaney',
            mel_scale='htk'
        )

        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    # 
    def forward(self, waveform: torch.Tensor, src_sr: Optional[int] = None) -> torch.Tensor:
        """
        Apply Log-Mel Spectrogram transformation to the input waveform.
        to_mono -> resample -> to_mel_spectrogram -> log_scale -> normalize_mel 

        Args:
            waveform (torch.Tensor): Input waveform tensor of shape (B, C, T) or (C, T).

        Returns:
            torch.Tensor: Transformed Log-Mel spectrogram tensor. The full pipeline will
            return Log-Mel Tensor of shape (B, 1, n_mels, frames).
        """
        x = waveform

        # Resampling (only when source sr provided and differs)
        if src_sr is not None:
            x = self.resample(x, src_sr)

        # To mono
        x = self.to_mono(x)

        # To Mel spectrogram (returns (1, n_mels, frames))
        x = self.to_mel_spectrogram(x)

        # Log scaling
        if self.log_mel:
            x = self.log_scale(x)

        # Normalization
        if self.normalize:
            x = self.normalize_mel(x)

        # Ensure final shape is (C=1, F, T)
        if x.dim() == 4 and x.shape[0] == 1:
            # (B=1, 1, F, T) -> (1, F, T)
            x = x.squeeze(0)
        return x
    

    # waveform mono 평균화 메서드, Channel 차원을 평균해 mono (channel=1)로 변환
    def to_mono(self, waveform : torch.Tensor) -> torch.Tensor:
        """
        Convert the multi-channel waveform to mono channel by averaging channel dimension.


        Args:
            waveform (torch.Tensor): Input waveform tensor of shape (B, C, T) or (C, T).

        Returns:
            torch.Tensor: Mono waveform tensor of shape (B, 1, T) or (1, T).

        Raises:
            ValueError: If the input waveform does not have 2 or 3 dimensions.
        
        
        
        """

        if waveform.dim() == 3:
            # (B, C, T) -> (B, 1, T)
            return waveform.mean(dim=1, keepdim=True)
        elif waveform.dim() == 2:
            # (C, T) -> (1, T)
            return waveform.mean(dim=0, keepdim=True)
        else:
            raise ValueError("Input waveform must have 2 or 3 dimensions.")



    # Resampling 메서드, LogMelSpectrogram 클래스 내의 sample_rate 속성에 맞게 리샘플링
    def resample(self, waveform : torch.Tensor, src_sr : int) -> torch.Tensor:
        """
        Resample the input waveform to the target sample rate defined in the class.

        Args:
            waveform (torch.Tensor): Input waveform tensor of shape (B, C, T) or (C, T).
            src_sr (int): Source sample rate of input waveform

        Returns:
            torch.Tensor: Resampled waveform tensor. Only time dimension changes.

        Raises:
            ValueError: If the input waveform does not have 2 or 3 dimensions.
        """

        # If source sample rate matches target, return original waveform
        if src_sr == self.sample_rate:
            return waveform  # No resampling needed
        
        #Create Resampler
        resampler = torchaudio.transforms.Resample(orig_freq=src_sr, new_freq=self.sample_rate)

        # Apply Resampling
        if waveform.dim() == 2:
            #
            return resampler(waveform)

        elif waveform.dim() == 3:
            # (B, C, T) -> (B*C, T)
            B, C, T = waveform.shape
            waveform = waveform.view(B * C, T)  # (B*C, T)
            resampled = resampler(waveform)  # (B*C, T')
            T_new = resampled.shape[1]
            return resampled.view(B, C, T_new)  # (B, C, T')


        else:
            raise ValueError("Input waveform must have 2 or 3 dimensions.")


    # Mel Spectrogram 변환 메서드
    def to_mel_spectrogram(self, waveform : torch.Tensor) -> torch.Tensor:
        """
        Convert mono waveform into Mel Spectrogram(linear power scale, before dB)


        Args:
            waveform (torch.Tensor): Input waveform tensor of shape (B, 1, T) or (1, T). 
            Use to_mono() method before if input is multi-channel.

        Returns:
            torch.Tensor: Output Mel Spectrogram tensor. 
                If input is (1, T): returns (1, 1, n_mels, frames)  [batch dim added]
                If input is (1, T), output is (1, n_mels, frames).

        Raises:
            ValueError: If the input waveform does not have 2 or 3 dimensions.
            
        """

    # Ensure mel_spectrogram is on the same device as waveform
        self.mel_spectrogram = self.mel_spectrogram.to(waveform.device)

        if waveform.dim() == 2:
            if waveform.size(0) != 1:
                raise ValueError("Input waveform must have shape (1, T) for mono audio.")
            # (1, T) -> (1, n_mels, frames)
            mel_spec = self.mel_spectrogram(waveform)  # (1, n_mels, frames)
            return mel_spec

        
        elif waveform.dim() == 3:
            if waveform.size(1) != 1:
                raise ValueError("Input waveform must have shape (B, 1, T) for mono audio.")

            mel_spec = self.mel_spectrogram(waveform)  # (B, 1, n_mels, frames)
            # For a single-sample path we keep batch; caller will squeeze to (1, F, T)
            return mel_spec



        else:
            raise ValueError("Input waveform must have 2 or 3 dimensions.")


    # Log Scaling 메서드
    def log_scale(self, mel_spectrogram : torch.Tensor) -> torch.Tensor:
        """
        Convert linear power Mel Spectrogram on dB Scale

        Args:
            mel_spectrogram (torch.Tensor): Input Mel Spectrogram tensor of shape 
            (B, 1, n_mels, frames)

        Returns:
            torch.Tensor: Log-scaled Mel Spectrogram tensor of the same shape as input.
            (B, 1, n_mels, frames)
        """

        # Ensure amp_to_db is on the same device as mel_spectrogram
        self.amp_to_db = self.amp_to_db.to(mel_spectrogram.device)

        # Apply Amplitude to dB conversion
        return self.amp_to_db(mel_spectrogram)



    # Normalize 메서드
    def normalize_mel(self, mel_spectrogram : torch.Tensor, eps : float = 1e-6) -> torch.Tensor:
        """
        Standardize Mel Spectrogram Tensor per sample across frequency bins and time frames.

        Args:
            mel_spectrogram (torch.Tensor): Input Mel Spectrogram tensor of shape
            (B, 1, n_mels, frames)

        Returns:
            torch.Tensor: Standardized Mel Spectrogram tensor of the same shape as input.
            (B, 1, n_mels, frames)
        """


        # Compute mean and std over (n_mels, frames) for each sample in batch
        mean = mel_spectrogram.mean(dim=(-2, -1), keepdim=True)
        std = mel_spectrogram.std(dim=(-2, -1), keepdim=True).clamp_min(eps)
        return (mel_spectrogram - mean) / std
        

