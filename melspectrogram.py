import librosa
import torch

class MelSpectrogram(torch.nn.Module):
    def __init__(self, sample_rate, frame_shift, frame_length, fft_length, num_mel_bins, freq_range):
        super().__init__()
        self.frame_shift = frame_shift
        self.frame_length = frame_length
        self.fft_length = fft_length
        fmin, fmax = freq_range if freq_range is not None else (None, None)
        self.register_buffer("mel_filter", torch.from_numpy(
            librosa.filters.mel(sr=sample_rate, n_fft=fft_length,
                                n_mels=num_mel_bins, fmin=fmin, fmax=fmax)).float()[None])
        self.register_buffer("window", torch.hann_window(self.frame_length))

    def forward(self, audio):
        """
        audio: [batch_size x num_samples]
        spec: [batch_size x num_mel_bins x num_frames]
        """
        pad_size = (self.fft_length - self.frame_shift) // 2
        audio = torch.nn.functional.pad(audio.unsqueeze(1), (pad_size, pad_size), mode="reflect").squeeze(1)  # [B x T]
        spec = torch.stft(audio, self.fft_length, hop_length=self.frame_shift, win_length=self.frame_length, window=self.window,
                          center=False, pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
        spec = torch.sqrt(spec.real**2 + spec.imag**2 + 1.0e-9)

        spec = self.mel_filter @ spec  # [B x M x T]
        spec = torch.log(spec.clamp(min=1.0e-5))
        return spec
