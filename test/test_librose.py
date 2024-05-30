import numpy as np
import librosa
import time

import scipy.signal
import matplotlib.pyplot as plt
from replay.preprocess.frequency import read_audio

def remove_background_noise(y, noise_sample, hop_length, noise_sample_duration=0.5):
    # Select a segment of the audio that contains only noise
    
    # Calculate the noise spectrum
    noise_stft = librosa.stft(noise_sample, n_fft=512, hop_length=hop_length)
    

    # Calculate the noise threshold
    noise_threshold = np.percentile(np.abs(noise_stft), 95, axis=1)
    
    # Perform spectral gating on the full audio
    stft = librosa.stft(y, n_fft=512, hop_length=hop_length)

    stft_filtered = np.abs(stft) - noise_threshold[:, np.newaxis]
    stft_filtered[stft_filtered < 0] = 0
    
    return stft_filtered

audio = read_audio(r"E:\behav\SMT\27049\20220516")
noise_sample = audio['audio'][int(43000 / audio['video_fps'] * audio['audio_fps']) : int(45500 / audio['video_fps'] * audio['audio_fps'])]
noise_stft = librosa.stft(noise_sample, n_fft=512, hop_length=int(audio['audio_fps'] / audio['video_fps'])+1)
noise_threshold = np.percentile(np.abs(noise_stft), 95, axis=1) #np.median(np.abs(noise_stft), axis=1)

import pickle
with open(r"E:\behav\background_noise.pkl", "wb") as f:
    pickle.dump(noise_threshold, f)

# Remove background noise
t1 = time.time()
stft_filtered = remove_background_noise(audio['audio'], noise_sample, hop_length=int(audio['audio_fps'] / audio['video_fps']))
print("time:", time.time() - t1)

print(stft_filtered.shape)
# Plot the filtered waveform]
power = np.abs(stft_filtered) ** 2

