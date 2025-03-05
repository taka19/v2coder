import numpy as np


def num_params(params):
    return sum([p.numel() for p in filter(lambda x: x.requires_grad, params)])

def load_wav(filename):
    from scipy.io.wavfile import read
    sr, data = read(filename)
    assert data.dtype == np.int16 and data.ndim == 1
    data = data.astype(np.float32) / 32768.0
    return data, sr

def save_wav(x, filename, sr):
    from scipy.io.wavfile import write
    assert x.dtype == np.float32 and x.ndim == 1
    data = np.clip(x * 32768.0, -32768.0, 32767.0)
    data = data.astype(np.int16)
    write(filename, sr, data)
