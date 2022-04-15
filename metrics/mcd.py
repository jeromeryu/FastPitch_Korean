import numpy as np
import librosa
from pysptk.conversion import sp2mc

from nnmnkwii.metrics import melcd
from nnmnkwii.preprocessing.alignment import DTWAligner

def main(target_path, pred_path, stft_params):
    """
    calculates mel-cepstrum distortion(MCD)
    """
    target_mc = wav2mc(target_path, stft_params) # (B, T1, C)
    pred_mc = wav2mc(pred_path, stft_params) # (B, T2, C)

    # Alignment - Dynamic Time Warping (DTW)
    aligner = DTWAligner()
    target_mc_aligned, pred_mc_aligned = aligner.transform((target_mc, pred_mc)) # (B, T3, C), (B, T3, C)

    mcd = melcd(pred_mc_aligned, target_mc_aligned)
    print('MCD', round(mcd, 4))

def wav2mc(wav_path, stft_params):
    """
    Computes mel-cepstrum
    """
    sp = wav2sp(wav_path, stft_params)
    mc = sp2mc(sp.T, 16, 0.49) # (T, C)
    
    mc = np.expand_dims(mc, axis=0) # (B, T, C)
    return mc

def wav2sp(wav_path, stft_params):
    """
    Computes power spectrogram
    """
    wav, sr = librosa.load(wav_path, sr=22050) 
    sp = np.abs(librosa.stft(wav, **stft_params)) ** 2 # (C, T)

    return sp

if __name__ == "__main__": 
    target_path = 'LJSpeech-1.1/wavs/LJ022-0023.wav'
    pred_path = 'output/audio_devset10_fp32_fastpitch_waveglow_denoise-0.01/001_the_overwhelming_majority_of_people_in.wav'

    stft_params = {}
    stft_params['n_fft'] = 1024
    stft_params['hop_length'] = stft_params['n_fft'] // 4
    stft_params['win_length'] = 1024

    main(target_path, pred_path, stft_params)