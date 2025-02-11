import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import matplotlib.pyplot as plt
import librosa

import torch
import torchaudio

from AudioEventDataset import AudioEventDataset

import numpy as np

def single_audio():

    fig, axs = plt.subplots(4, 1, figsize = (20, 20))

    audio_file, _ = librosa.load("/home/ldomeneghetti/Documents/Forensics/audio_classifier_pytorch/custom_audio/car_crash.wav")

    axs[0].plot(audio_file)
    axs[0].set_title("Waveform")

    stft = librosa.stft(audio_file, n_fft=2048, hop_length=512)
    spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    axs[1].imshow(spectrogram, cmap='hot', origin='lower', aspect='auto')
    axs[1].set_title("Short-Time Fourier Transform")
    axs[1].set_ylabel('Frequency')

    mel_spectrogram = librosa.feature.melspectrogram(y=audio_file, sr=44100, n_fft=2048, hop_length=512)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    librosa.display.specshow(log_mel_spectrogram, sr=44100, hop_length=512, y_axis='mel', ax = axs[2])
    axs[2].set_title("Mel Spectrogram")
    axs[2].set_ylabel('Mel Frequency')

    mfccs = librosa.feature.mfcc(y=audio_file, sr=44100, n_fft=2048, hop_length=512, n_mfcc=40)
    librosa.display.specshow(mfccs, sr=44100, hop_length=512)
    axs[3].set_title("Mel Frequency Cepstral Coefficients")
    axs[3].set_ylabel('MFCC Coefficient')

    plt.tight_layout()
    plt.show()

def framed_plot():

    audio_data, _ = torchaudio.load("/home/ldomeneghetti/Documents/Forensics/audio_classifier_pytorch/raw_audio/gun_shot/122690-6-0.wav", normalize = True)
    framed_audio_data = AudioEventDataset.frame_audio_overlap(audio_data, 1, 44100, 0.5)

    fig, axs = plt.subplots(4, framed_audio_data.size()[0], figsize = (20, 20))

    # framed_audio_data = framed_audio_data.tolist()

    for index, frame in enumerate(framed_audio_data):
        axs[0][index].set_ylim([-1, 1])
        axs[0][index].plot(frame)

        frame = frame.numpy()

        stft = librosa.stft(frame, n_fft=2048, hop_length=512)
        spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        axs[1][index].imshow(spectrogram, cmap='hot', origin='lower', aspect='auto')

        mel_spectrogram = librosa.feature.melspectrogram(y=frame, sr=44100, n_fft=2048, hop_length=512)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        librosa.display.specshow(log_mel_spectrogram, sr=44100, hop_length=512, y_axis='mel', ax = axs[2][index])

        mfccs = librosa.feature.mfcc(y=frame, sr=44100, n_fft=2048, hop_length=512, n_mfcc=40)
        librosa.display.specshow(mfccs, sr=44100, hop_length=512, ax = axs[3][index])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    framed_plot()
