import numpy as np
import pandas as pd
import time
import pickle
import os

from replay.preprocess.behav import read_dlc, process_dlc, get_reward_info
from replay.preprocess.frequency import read_audio, sliding_stft
from replay.preprocess.frequency import remove_background_noise
from replay.preprocess.frequency import get_dominant_frequency


def run_all_sessions(
    f: pd.DataFrame,
    i: int
):
    # Initialize basic information
    trace = {
        'mouse': f['MiceID'][i],
        'date': f['date'][i],
        'session': f['session'][i],
        'paradigm': f['paradigm'][i], 
    }
    print(
        f"{i}  Mouse {trace['mouse']} {trace['date']} {trace['paradigm']}"
        f"-----------------------"
    )
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    # Read Background Noise
    with open(f['background_noise'][i], 'rb') as handle:
        background_noise = pickle.load(handle)

    # Read audio
    print("  1. Read Audio")
    audio = read_audio(f['recording_folder'][i])
    print(f"     Info: {audio['duration']} s, ")
    trace.update(audio)

    # Short-time Fourier Transform to get real-time frequency.
    print("  2. Short-time Fourier Transform")
    frequencies, magnitudes = sliding_stft(
        audio['audio'], 
        duration = audio['duration'],
        targ_frames = audio['video_frames'],
        n = 512
    )
    print("      Quality Control - Remove Background Noise")
    magnitudes = remove_background_noise(magnitudes, background_noise)
    power = magnitudes ** 2
    behav_freq = get_dominant_frequency(magnitudes, frequencies)
    trace.update({
        'frequencies': frequencies, 
        'magnitudes': magnitudes,
        'behav_freq': behav_freq,
        'power': power
    })

    # Get reward info
    print("  3. Get Lever Status")
    dlc_data = read_dlc(f['recording_folder'][i])
    trace.update(process_dlc(dlc_data))

    # Get reward info
    print("  4. Get Reward Status")
    reward_time, reward_label = get_reward_info(f['recording_folder'][i])
    trace['reward_time'] = reward_time
    trace['reward_label'] = reward_label

    print("  5. Save Data")
    file_dir = os.path.join(f['recording_folder'][i], 'trace_behav.pkl')
    with open(file_dir, 'wb') as handle:
        pickle.dump(trace, handle)

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print("Done.", end='\n\n')


if __name__ == "__main__":
    from local_path import f1_behav

    run_all_sessions(f1_behav, 0)