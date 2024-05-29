from moviepy.editor import VideoFileClip, AudioFileClip
import numpy as np
import os
from scipy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
from tqdm import tqdm

def read_audio(
    dir_name: str, 
    file_name: str = "video.mp4"
) -> dict:
    """Read the audio from behavioral video"""
    f_dir = os.path.join(dir_name, file_name)

    # Read audio from video
    video = VideoFileClip(f_dir)
    audio = video.audio

    print(video.duration, video.duration*video.fps, audio.fps)

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

def fourier_transform(
    audio_array: np.ndarray,
    fps: int = 44100,
    n: int = 4096,
    fft_kwargs: dict = {}
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert single-channel audio to frequency information using FFT.

    Parameters
    ----------
    audio_array : np.ndarray
        A 1D numpy array containing the audio samples from the single-channel
          audio.
    fps : int
        The frames per second (sampling rate) of the audio in Hz (e.g., 44100
          for CD-quality audio).
    n : int, optional
        The number of time bins for FFT. By default 2048.
    fft_kwargs : dict
        Additional keyword arguments to pass to the `fft` function.

    Returns
    -------
    frequencies : np.ndarray
        A 1D numpy array containing the positive frequency components in Hz.
    magnitudes : np.ndarray
        A 1D numpy array containing the magnitude of each frequency component.
    
    Notes
    -----
    The Fast Fourier Transform (FFT) is used to convert the time-domain audio
      signal into the frequency domain. The magnitude of the FFT represents 
      the amplitude of each frequency component present in the audio signal.
    
    Example
    -------
    >>> audio_array = np.array([0.0, 0.1, 0.0, -0.1])
    >>> fps = 44100
    >>> frequencies, magnitudes = fourier_transform(audio_array, fps)
    >>> print(frequencies)
    [0.0, 11025.0, 22050.0]
    >>> print(magnitudes)
    [0.0, 0.1, 0.2]
    """
    # Perform FFT
    yf = rfft(audio_array, n = n, **fft_kwargs)
    xf = rfftfreq(n, 1 / fps)
    return xf, yf

def stft(
    audio: np.ndarray,
    duration: float,
    targ_frames: int,
    fps: int = 44100,
    n: int = 4096,
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
    fps : int
        The frames per second (sampling rate) of the audio in Hz.

    Returns
    -------
    times : np.ndarray
        A 1D numpy array containing the time points for each window in seconds.
    frequencies : np.ndarray
        A 1D numpy array containing the frequency components in Hz.
    magnitudes : np.ndarray
        A 2D numpy array where each row corresponds to the magnitude of each 
        frequency component for a given window.
    """
    N = int(n / 2) + 1
    audio_time = np.linspace(0, duration, audio.shape[0]+1)[:-1]
    downsampled_time = np.linspace(0, duration, targ_frames+1)
    magnitudes = np.zeros((N, targ_frames), dtype=np.float64)
    frequencies = rfftfreq(n, 1 / fps)

    print("Short-time Fast Fourier Transformation Started:")
    for i in tqdm(range(targ_frames)):
        idx = np.where(
            (downsampled_time[i] <= audio_time) & 
            (audio_time < downsampled_time[i+1])
        )[0]
        windowed_signal = audio[idx]

        # Perform FFT
        magnitudes[:, i] = np.abs(rfft(windowed_signal, n = n, **fft_kwargs))

    return downsampled_time, frequencies, magnitudes

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
    times : np.ndarray
        A 1D numpy array containing the time points for each window in seconds.
    frequencies : np.ndarray
        A 1D numpy array containing the frequency components in Hz.
    magnitudes : np.ndarray
        A 2D numpy array where each row corresponds to the magnitude of each 
        frequency component for a given window.
    """
    N = int(n / 2) + 1
    audio_time = np.linspace(0, duration, audio.shape[0]+1)[:-1]
    downsampled_time = np.linspace(0, duration, targ_frames+1)
    magnitudes = np.zeros((N, targ_frames), dtype=np.float64)
    frequencies = rfftfreq(n, 1 / fps)

    print("Short-time Fast Fourier Transformation Started:")
    for i in tqdm(range(targ_frames)):
        idx = np.where(
            (downsampled_time[i] - t_window <= audio_time) & 
            (audio_time < downsampled_time[i] + t_window)
        )[0]
        windowed_signal = audio[idx]

        # Perform FFT
        magnitudes[:, i] = np.abs(rfft(windowed_signal, n = n, **fft_kwargs))

    return downsampled_time[:-1], frequencies, magnitudes

if __name__ == "__main__":
    import pickle
    
    dir_name = r"E:\behav\SMT\27049\20220516"
    
    audio = read_audio(dir_name)
    """
    downsampled_time, frequencies, magnitudes = sliding_stft(
        audio['audio'], 
        duration = audio['duration'],# 1800.19, 
        targ_frames = audio['video_frames'], #54005,
        n = 512
    )
    dominant_freq = frequencies[np.argmax(magnitudes, axis=0)]

    with open(os.path.join(dir_name, r"temp3.pkl"), "wb") as f:
        pickle.dump([downsampled_time, frequencies, magnitudes], f)
    """
    with open(os.path.join(dir_name, r"temp3.pkl"), "rb") as f:
        downsampled_time, frequencies, magnitudes = pickle.load(f)

    print(magnitudes.shape, frequencies.shape)

    magnitudes = np.abs(magnitudes)
    magnitudes = magnitudes / np.max(magnitudes, axis=0)    
    power = magnitudes ** 2
    power = power[frequencies > 2000, :]
    frequencies = frequencies[frequencies > 2000]
    dominant_freq = frequencies[np.argmax(power, axis=0)]
    
    print(np.min(magnitudes), np.max(magnitudes), frequencies.shape)

    from replay.preprocess.behav import read_dlc, process_dlc, get_reward_info
    dir_name = r"E:\behav\SMT\27049\20220516"
    dlc_data = read_dlc(dir_name)
    trace = process_dlc(dlc_data)       

    is_press = trace['is_press']
    time = audio['video_time']

    # reward
    reward_time1 = get_reward_info(dir_name, '液体奖励')
    reward_time2 = get_reward_info(dir_name, '液体输出')
    reward_time3 = get_reward_info(dir_name, '饮水')

    plt.plot(reward_time1, np.repeat(1600, len(reward_time1)), 'o')
    plt.plot(reward_time2, np.repeat(1400, len(reward_time2)), 'o')
    plt.plot(reward_time3, np.repeat(1200, len(reward_time3)), 'o')
    print(is_press, np.sum(is_press))
    plt.plot(time[np.where(is_press == 1)[0]], np.repeat(1800, np.sum(is_press)), 'o')
    plt.plot(time, dominant_freq[:-1])
    plt.show()
