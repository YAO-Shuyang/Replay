from moviepy.editor import VideoFileClip, AudioFileClip
import numpy as np
import os
import librosa
from typing import Optional
from scipy.fft import rfftfreq
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from tqdm import tqdm
import copy as cp

def read_audio(
    dir_name: str, 
    file_name: str = "video.mp4"
) -> dict:
    """Read the audio from behavioral video"""
    f_dir = os.path.join(dir_name, file_name)

    if os.path.exists(f_dir) == False:
        file_name = "video.avi"
        f_dir = os.path.join(dir_name, file_name)

    # Read audio from video
    video = VideoFileClip(f_dir)
    audio = video.audio

    # Convert to array
    audio_array = audio.to_soundarray()
    n_frame = audio_array.shape[0]
    duration = video.duration
    video_fps = video.fps
    video_frames = int(duration * video_fps)
    audio_fps = audio.fps
    
    video.close()
    # Convert to single-channel
    audio_array = np.mean(audio_array, axis=1)

    # Get time from duration and convert it to milisecond
    audio_time = np.linspace(0, audio.duration, n_frame+1)[:-1]
    video_time = np.linspace(0, duration, video_frames)[:-1]

    return {
        "duration": duration,
        "audio": audio_array,
        "audio_time": audio_time,
        "audio_fps": audio_fps,
        "audio_frames": n_frame,
        "video_time": video_time,
        "video_fps": video_fps,
        "video_frames": video_frames
    }

def sliding_stft(
    audio: np.ndarray,
    duration: float,
    targ_frames: int,
    t_window: int = 0.1,
    fps: int = 44100,
    n: int = 512,
    fft_kwargs: dict = {}
) -> np.ndarray:
    """
    Perform Short-Time Fourier Transform (STFT) on the audio array.

    Parameters
    ----------
    audio_array : np.ndarray
        An 1D numpy array containing the audio samples from the single-channel
        audio.
    duration : float
        The duration of the audio in seconds.
    targ_frames : int
        The number of time bins for STFT.
    t_window : int
        The length of the window in samples, in seconds.
    fps : int
        The frames per second (sampling rate) of the audio in Hz.
    n : int
        The number of frequency bins for FFT.

    Returns
    -------
    frequencies : np.ndarray
        A 1D numpy array containing the frequency components in Hz.
    magnitudes : np.ndarray
        A 2D numpy array where each row corresponds to the magnitude of each 
        frequency component for a given window.
    """
    N = int(n / 2) + 1
    frequencies = rfftfreq(n, 1 / fps)

    magnitudes = librosa.stft(
        audio, 
        n_fft=n, 
        hop_length=int(audio.shape[0] / targ_frames), 
        **fft_kwargs
    )
    magnitudes = np.abs(magnitudes)
    return frequencies, magnitudes[:, :targ_frames]

def get_background_noise(
    magnitudes: np.ndarray,
    range: tuple[int, int]
) -> np.ndarray:
    """
    Input a piece of pure background noise
    """
    noise_magnitudes = np.abs(magnitudes[:, range[0]:range[1]])
    noise_magnitudes = np.percentile(noise_magnitudes, 95, axis=1)
    return noise_magnitudes

def remove_background_noise(
    magnitudes: np.ndarray, 
    background_noise: np.ndarray
) -> np.ndarray:
    """
    Remove the background noise from the audio signal.

    Notes
    -----
    Background noise was generated in advance.
    """
    magnitudes[23:, :] = magnitudes[23:, :] - background_noise[23:, np.newaxis]
    magnitudes[:23, :] *= 0.0001
    magnitudes[magnitudes < 0] = 0

    # Normalize across axis 0
    magnitudes = magnitudes / np.max(magnitudes, axis=0)
    return magnitudes

def get_dominant_frequency(magnitudes: np.ndarray) -> np.ndarray:
    """
    Get the dominant frequency from the audio signal.
    """
    return np.argmax(magnitudes, axis=0)

def correct_freq(
    dominant_freq: np.ndarray,
    magnitudes: np.ndarray,
    onset_frame: int,
    end_frame: int
) -> np.ndarray:
    
    for i in range(onset_frame, end_frame):
        curr_f = dominant_freq[i]
        next_f = np.argmax(magnitudes[curr_f:180, i+1]) + curr_f
        next_next_f = np.argmax(magnitudes[curr_f:180, i+2]) + curr_f
        if next_f < curr_f:
            dominant_freq[i+1] = curr_f
        elif next_f > 1.8 * curr_f: 
            dominant_freq[i+1] = curr_f
        else:
            if np.abs(next_f - next_next_f) <= 2 or i == end_frame-1:
                dominant_freq[i+1] = next_f
            else:
                dominant_freq[i+1] = curr_f
    
    return dominant_freq

def reset_freq(
    dominant_freq: np.ndarray,
    onset_frame: int,
    end_frame: int
) -> np.ndarray:
    dominant_freq[onset_frame:end_frame+1] = 0
    return dominant_freq

def display_spectrum(
    ax: Axes,
    magnitudes: np.ndarray,
    freq_range: tuple[int, int],
    frame_range: tuple[int, int],
    dominant_freq: Optional[np.ndarray] = None
) -> Axes:
    magnitudes /= np.max(magnitudes, axis=0)

    if dominant_freq is not None:
        ax.plot(np.arange(magnitudes.shape[1]), dominant_freq, color = 'yellow')

    ax.imshow(magnitudes, aspect="auto", cmap='hot', interpolation='nearest')
    ax.set_xlim(frame_range)
    ax.set_ylim(freq_range)
    return ax

def filter_freq(
    dominant_freq: np.ndarray,
    magnitudes: np.ndarray,
    onsets: np.ndarray,
    ends: np.ndarray
) -> np.ndarray:
    assert len(onsets) == len(ends)
    dominant_freq_filtered = np.zeros_like(dominant_freq)
    for i in range(len(onsets)):
        dominant_freq_filtered[onsets[i]] = np.argmax(
            magnitudes[:, onsets[i]]
        )
        dominant_freq_filtered = correct_freq(
            dominant_freq_filtered, 
            magnitudes, 
            onsets[i], 
            ends[i]
        )

    return dominant_freq_filtered

def update_end_freq(
    end_freq: np.ndarray, 
    dominant_freq: np.ndarray,
    ends: np.ndarray
) -> np.ndarray:
    assert len(ends) == len(end_freq)

    for i in range(len(ends)):
        end_freq[i] = dominant_freq[ends[i]]

    return end_freq

if __name__ == "__main__":
    import pickle
    from replay.preprocess.behav import (
        read_dlc, process_dlc, get_reward_info, get_lever_event, 
        coordinate_event_behav
    )
    from replay.preprocess.behav import identify_trials, get_freqseq_info
    dir_name = r"E:\behav\SMT\27049\20220516\session 1"
    
    audio = read_audio(dir_name)

    audio_conv = audio['audio']

    frequencies, magnitudes = sliding_stft(
        audio_conv, 
        duration = audio['duration'],# 1800.19, 
        targ_frames = audio['video_frames'], #54005,
        n = 512
    )
    ori_magnitudes = cp.deepcopy(magnitudes)
    
    ratio = audio['audio_fps'] / audio['video_fps']
    background_noise = get_background_noise(
        magnitudes, 
        range = (41500, 46100),#(int(31200 * ratio), int(33700 * ratio)), 
    )
    
    magnitudes = remove_background_noise(magnitudes, background_noise)
    final_freq_time, dominant_freq = get_freqseq_info(dir_name, frequencies=frequencies)
    time_indicator = coordinate_event_behav(final_freq_time, audio['video_time'])

    dominant_freq = get_dominant_frequency(magnitudes)
    magnitudes = magnitudes / np.max(magnitudes, axis=0)

    
    releasing_time = get_lever_event(dir_name)
    is_releasing = coordinate_event_behav(releasing_time, audio['video_time'])

    lever = process_dlc(read_dlc(dir_name))
    is_press = lever['is_press']

    # reward
    #reward_time = get_reward_info(dir_name)
    print(time_indicator, time_indicator.shape)
    onset, end, dominant_freq_filtered, end_freq = identify_trials(
        dominant_freq=cp.deepcopy(dominant_freq)
    )
    print(onset)
    print(end)
    idx = np.where(dominant_freq_filtered != 0)[0]
    
    print(onset.shape)
    plt.figure()
    ax = plt.axes()
    plt.imshow(magnitudes, cmap='hot', interpolation='nearest')
    x = np.arange(magnitudes.shape[1])
    #plt.plot(x, dominant_freq , color = 'green', alpha = .5)
    idx = np.where(is_releasing == 1)[0]
    plt.plot(x[idx], is_releasing[idx]-2,'o', markeredgewidth = 0, markersize = 6)
    dp = np.ediff1d(is_press)
    idx = np.where(dp == -1)[0]
    plt.plot(x[idx], is_press[idx], 'o', markeredgewidth = 0, markersize = 6)
    plt.plot(x, dominant_freq_filtered , color = 'yellow')
    #plt.plot(time, dominant_freq[:-1])
    ax.set_aspect('auto')
    plt.show()
