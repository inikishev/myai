import numpy as np
import pedalboard
import torch

def audioread(path) -> tuple[np.ndarray, int]:
    """Returns audio and sr"""
    with pedalboard.io.AudioFile(path, 'r') as f: # pylint:disable=E1129 # type:ignore
        audio = f.read(f.frames)
        sr = f.samplerate
    return audio, sr

def audioreadtensor(path) -> tuple[torch.Tensor, int]:
    """Returns audio and sr"""
    audio, sr = audioread(path)
    return torch.from_numpy(audio), sr