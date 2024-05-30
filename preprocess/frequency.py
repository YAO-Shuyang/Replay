from moviepy.editor import VideoFileClip, AudioFileClip
import numpy as np
import os
import librosa
from scipy.fft import rfftfreq
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    magnitudes = magnitudes - background_noise[:, np.newaxis]
    magnitudes[magnitudes < 0] = 0

    # Normalize across axis 0
    magnitudes = magnitudes / np.max(magnitudes, axis=0)
    return magnitudes

def get_dominant_frequency(magnitudes: np.ndarray) -> np.ndarray:
    """
    Get the dominant frequency from the audio signal.
    """
    return np.argmax(magnitudes, axis=0)+1


if __name__ == "__main__":
    import pickle
    
    dir_name = r"E:\behav\SMT\27049\20220516"
    
    audio = read_audio(dir_name)

    frequencies, magnitudes = sliding_stft(
        audio['audio'], 
        duration = audio['duration'],# 1800.19, 
        targ_frames = audio['video_frames'], #54005,
        n = 512
    )

    with open(r"E:\behav\background_noise.pkl", "rb") as f:
        background_noise = pickle.load(f)

    #magnitudes = remove_background_noise(magnitudes, background_noise)
    dominant_freq = get_dominant_frequency(magnitudes)
    dominant_freq[dominant_freq <= 23] = 0

    from replay.preprocess.behav import read_dlc, process_dlc, get_reward_info
    dir_name = r"E:\behav\SMT\27049\20220516"
    dlc_data = read_dlc(dir_name)
    trace = process_dlc(dlc_data)

    is_press = trace['is_press']
    time = audio['video_time']

    # reward
    reward_time = get_reward_info(dir_name)

    plt.figure()
    ax = plt.axes()
    plt.imshow(magnitudes, cmap='hot', interpolation='nearest')
    plt.plot(reward_time, np.repeat(300, len(reward_time)), 'o')
    x = np.arange(magnitudes.shape[1])
    plt.plot(x, dominant_freq-1 , color = 'yellow')
    plt.plot(time[np.where(is_press == 1)[0]], np.repeat(280, np.sum(is_press)), 'o')
    #plt.plot(time, dominant_freq[:-1])
    ax.set_aspect('auto')
    plt.show()
