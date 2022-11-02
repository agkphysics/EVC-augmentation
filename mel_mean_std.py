import sys
import warnings

import librosa
import numpy as np
from tqdm.contrib.concurrent import process_map

warnings.filterwarnings("ignore", category=UserWarning, message="PySoundFile")


def extract_mel_spec(filename):
    '''
    extract and save both log-linear and log-Mel spectrograms.
    saved spec shape [n_frames, 1025]
    saved mel shape [n_frames, 80]
    '''
    y, sample_rate = librosa.load(filename,sr=16000)
    y = librosa.util.normalize(y)
    y, _ = librosa.effects.trim(y, top_db=20)
    spec = librosa.core.stft(
        y=y,
        n_fft=2048,
        hop_length=200,
        win_length=800,
        window='hann',
        center=True,
        pad_mode='reflect'
    )
    spec, _ = librosa.magphase(spec)
    mel = librosa.feature.melspectrogram(
        S=spec,
        sr=sample_rate,
        n_mels=80,
        power=1.0, #actually not used given "S=spec"
        fmin=0.0,
        fmax=None,
        htk=False,
        norm=1
    )
    mel = np.log(np.maximum(mel, 1e-5)).astype(np.float32)
    return mel.T


def main():
    with open(sys.argv[1]) as fid:
        abs_paths = list(map(str.strip, fid))

    mels = process_map(extract_mel_spec, abs_paths, chunksize=1)
    mels = np.vstack(mels)

    mel_mean = np.mean(mels, axis=0)
    mel_std = np.std(mels, axis=0)
    np.save(sys.argv[2], [mel_mean, mel_std])


if __name__ == "__main__":
    main()
