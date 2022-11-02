import warnings
from typing import Tuple

import librosa
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, message="PySoundFile")


def extract(path, trim: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Extract log STFT and log mel spectrograms.

    Returns
    -------
    mel: np.ndarray
        The log mel spectrogram of shape (t, n_mels)
    spec: np.ndarray
        The STFT of shape (t, n_fft // 2 + 1)
    """
    audio, sr = librosa.load(path, sr=16000)
    if trim:
        audio, _ = librosa.effects.trim(audio, top_db=20)
    spec = librosa.stft(
        audio,
        n_fft=2048,
        hop_length=200,
        win_length=800,
        window="hann",
        center=True,
        pad_mode="reflect",
    )
    spec, _ = librosa.magphase(spec)
    mel = librosa.feature.melspectrogram(
        S=spec,
        sr=sr,
        n_mels=80,
        power=1.0,  # actually not used given "S=spec"
        fmin=0.0,
        fmax=None,
        htk=False,
        norm=1,
    )
    spec = np.log(np.maximum(spec, 1e-10)).astype(np.float32)
    mel = np.log(np.maximum(mel, 1e-10)).astype(np.float32)
    spec = spec.T
    mel = mel.T
    return mel, spec
