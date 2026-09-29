import numpy as np
import scipy.io.wavfile as wavfile


def read_wav(path):
    """Mono float signal in [-1, 1] and its sample rate.

    16-bit samples are divided by 2^15 rather than by the file's own peak, so
    levels stay comparable between files and match what is written back.
    """
    rate, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    if np.issubdtype(data.dtype, np.integer):
        data = data / float(np.iinfo(data.dtype).max + 1)
    return data.astype(np.float64), rate


def write_wav(path, signal, rate):
    pcm = np.clip(np.round(signal * 32768), -32768, 32767).astype(np.int16)
    wavfile.write(path, rate, pcm)
