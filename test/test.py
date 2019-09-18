import wave
import numpy as np
import torch
import scipy.io as sio
import scipy.io.wavfile

from collections import namedtuple
import random

import torch
import torchaudio
from torchaudio import transforms

def _wav2array(nchannels, sampwidth, data):
    """data must be the string containing the bytes from the wav file."""
    num_samples, remainder = divmod(len(data), sampwidth * nchannels)
    if remainder > 0:
        raise ValueError('The length of data is not a multiple of '
                         'sampwidth * num_channels.')
    if sampwidth > 4:
        raise ValueError("sampwidth must not be greater than 4.")

    if sampwidth == 3:
        a = np.empty((num_samples, nchannels, 4), dtype=np.uint8)
        raw_bytes = np.fromstring(data, dtype=np.uint8)
        a[:, :, :sampwidth] = raw_bytes.reshape(-1, nchannels, sampwidth)
        a[:, :, sampwidth:] = (a[:, :, sampwidth - 1:sampwidth] >> 7) * 255
        result = a.view('<i4').reshape(a.shape[:-1])
    else:
        # 8 bit samples are stored as unsigned ints; others as signed ints.
        dt_char = 'u' if sampwidth == 1 else 'i'
        a = np.fromstring(data, dtype='<%s%d' % (dt_char, sampwidth))
        result = a.reshape(-1, nchannels)
    return result

def readwav(file):
    """
    Read a wav file.
    Returns the frame rate, sample width (in bytes) and a numpy array
    containing the data.
    This function does not read compressed wav files.
    """

    sample_rate, data = sio.wavfile.read(file)
    wav = wave.open(file)
    rate = wav.getframerate()
    nchannels = wav.getnchannels()
    sampwidth = wav.getsampwidth()
    nframes = wav.getnframes()
    data = wav.readframes(nframes)
    wav.close()
    array = _wav2array(nchannels, sampwidth, data)

    print(type(array))
    print(array)
    print(array.shape)
    print(array.ravel())
    print(sample_rate)
    float_tens = torch.FloatTensor(array)

    print("===========")
    N_FFT = 1024
    win_length = N_FFT/sample_rate

    return rate, sampwidth, array

def tfm_spectro(file, sr=16000, to_db_scale=False, n_fft=1024, 
                ws=None, hop=None, f_min=0.0, f_max=-80, pad=0, n_mels=128):
    
    AudioData = namedtuple('AudioData', ['sig', 'sr'])
    audio = AudioData(*torchaudio.load(file))
    # We must reshape signal for torchaudio to generate the spectrogram.
    sr = audio.sr
    mel = transforms.MelSpectrogram(sample_rate=audio.sr, n_mels=n_mels, n_fft=n_fft, win_length=ws, hop_length=hop, 
                                    f_min=f_min, f_max=f_max, pad=pad,)(audio.sig.reshape(1, -1))
    mel = mel.permute(0,2,1) # swap dimension, mostly to look sane to a human.
    if to_db_scale: mel = transforms.SpectrogramToDB(stype='magnitude', top_db=f_max)(mel)
    print(mel)
    return mel


def get_spectrogram_feature(filepath):
    (rate, width, sig) = readwav(filepath)
    N_FFT = 1024
    sig = sig.ravel()

    #Get Sample rate from wave. 
    SAMPLE_RATE = 16000


    stft = torch.stft(torch.FloatTensor(sig),
                        N_FFT,
                        hop_length=int(0.01*SAMPLE_RATE),
                        win_length=int(0.030*SAMPLE_RATE),
                        window=torch.hamming_window(int(0.030*SAMPLE_RATE)),
                        center=False,
                        normalized=False,
                        onesided=True)

    stft = (stft[:,:,0].pow(2) + stft[:,:,1].pow(2)).pow(0.5);
    amag = stft.numpy();
    feat = torch.FloatTensor(amag)
    feat = torch.FloatTensor(feat).transpose(0, 1)

    return feat


if __name__ == "__main__":
    file_path = "../speech_hackathon_2019/sample_dataset/train/train_data/wav_002.wav"
        
    tfm_spectro(file_path, ws=512, hop=256, n_mels=128, to_db_scale=True, f_max=8000, f_min=-80.0)
    readwav(file_path)

