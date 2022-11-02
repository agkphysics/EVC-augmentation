import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt

import numpy as np
import librosa
import scipy.io.wavfile
import soundfile as sf

def plot_data(data, fn, figsize=(12, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        if len(data) == 1:
            ax = axes
        else:
            ax = axes[i]
        g = ax.imshow(data[i], aspect='auto', origin='lower',
                       interpolation='none')
        plt.colorbar(g, ax=ax)
    plt.savefig(fn)



def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = list(range(len(s1) + 1))
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def log_mel(post, mel_mean_std):
    mean, std = np.load(mel_mean_std)
    mel = post * std[:, None] + mean[:, None]
    return mel


def linear_mel(mel, mel_mean_std):
    mel = np.exp(log_mel(mel, mel_mean_std))
    return mel


def recover_wav(mel, wav_path, n_fft=2048, win_length=800, hop_length=200):
    filters = librosa.filters.mel(sr=16000, n_fft=n_fft, n_mels=80, norm=1)
    inv_filters = np.linalg.pinv(filters)
    spec = np.dot(inv_filters, mel)

    y = librosa.griffinlim(spec, n_iter=50, hop_length=hop_length, win_length=win_length, n_fft=n_fft, pad_mode="reflect")
    sf.write(wav_path, y, 16000, "PCM_16")
