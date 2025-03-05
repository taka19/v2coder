import librosa
import os
import torch

from melspectrogram import MelSpectrogram
import utils


def get_file_list(data_root, data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        file_list = [os.path.join(data_root, line.strip() + ".wav") for line in f]
    return file_list

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        audio, _ = utils.load_wav(filepath)
        audio = librosa.util.normalize(audio) * 0.95
        audio = torch.FloatTensor(audio)
        return audio

    def __len__(self):
        return len(self.file_list)

class AudioSegmentDataset(AudioDataset):
    def __init__(self, file_list, segment_size):
        super().__init__(file_list)
        self.segment_size = segment_size

    def __getitem__(self, idx):
        audio = super().__getitem__(idx)

        num_samples = len(audio)
        if num_samples < self.segment_size:
            audio = torch.cat([audio, audio.new_zeros([self.segment_size - num_samples])], dim=0)
            start = torch.tensor([0])
        else:
            start = torch.randint(num_samples - self.segment_size, [])

        audio = audio[start:start+self.segment_size]
        return audio

class Collator:
    def __init__(self, frame_shift, frame_length, fft_length, **kwargs):
        self.frame_shift = frame_shift
        self.frame_length = frame_length
        self.fft_length = fft_length
        self.melspectrogram = MelSpectrogram(
            frame_shift=frame_shift,
            frame_length=frame_length,
            fft_length=fft_length,
            **kwargs,
        )

    def __call__(self, batch):
        wav = torch.stack(batch, dim=0)
        wav.add_(torch.randn_like(wav) * 2.0**-16)
        msg = self.melspectrogram(wav)
        return wav[:, None], msg
