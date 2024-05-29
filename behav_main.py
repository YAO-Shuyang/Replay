import numpy as np
import pandas as pd
import time
import pickle
import os

from replay.preprocess.behav import read_dlc, process_dlc, get_reward_info
from replay.preprocess.frequency import read_audio, sliding_stft


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

    # Read audio
    print("  1. Read Audio")
    audio = read_audio(f['recording_folder'][i])
    trace.update(audio)

    # Short-time Fourier Transform to get real-time frequency.
    print("  2. Short-time Fourier Transform:")
    downsampled_time, frequencies, magnitudes = sliding_stft(
        audio['audio'], 
        duration = audio['duration'],
        targ_frames = audio['video_frames'],
        n = 512
    )
    trace.update({ 'frequencies': frequencies, 'magnitudes': magnitudes})

    # Get reward info
    print("  3. Get Lever Status")
    dlc_data = read_dlc(f['recording_folder'][i])
    trace.update(process_dlc(dlc_data))

    # Get reward info
    print("  4. Get Reward Status")
    reward_time = get_reward_info(f['recording_folder'][i])
    trace['reward_time'] = reward_time

    print("  5. Save Data")
    file_dir = os.path.join(f['recording_folder'][i], 'trace_behav.pkl')
    with open(file_dir, 'wb') as handle:
        pickle.dump(trace, handle)

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print("Done.", end='\n\n')


if __name__ == "__main__":
    from local_path import f1