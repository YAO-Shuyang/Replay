import numpy as np
import pandas as pd
import time
import pickle
import os

from replay.preprocess.behav import read_dlc, process_dlc, get_reward_info
from replay.preprocess.behav import coordinate_event_behav, identify_trials
from replay.preprocess.frequency import read_audio, sliding_stft
from replay.preprocess.frequency import remove_background_noise
from replay.preprocess.frequency import get_dominant_frequency


def run_section_one(
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
    print(f"     Info: {audio['duration']} s")
    print(f"     Video Fps: {audio['video_fps']}")
    print(f"     Video Frames: {audio['video_frames']}")
    print(f"     Audio Fps: {audio['audio_fps']}")
    trace.update(audio)

    # Short-time Fourier Transform to get real-time frequency.
    print("  2. Short-time Fourier Transform")
    frequencies, magnitudes = sliding_stft(
        audio['audio'], 
        duration = audio['duration'],
        targ_frames = audio['video_frames'],
        n = 512
    )

    trace.update({'frequencies': frequencies, 'magnitudes': magnitudes})
    trace['save_dir'] = os.path.join(f['recording_folder'][i], 'trace_behav.pkl')
    return trace

def run_section_two(trace: dict) -> dict:
    # Remove background noise
    print("      Quality Control - Remove Background Noise")
    trace['magnitudes'] = remove_background_noise(
        trace['magnitudes'], 
        trace['background_noise']
    )
    power = trace['magnitudes'] ** 2
    behav_freq = get_dominant_frequency(trace['magnitudes'])
    
    # Identify Trials
    print()
    trace.update({'behav_freq': behav_freq, 'power': power})

    # Get reward info
    print("  3. Get Lever Status")
    try:
        dlc_data = read_dlc(os.path.dirname(trace['save_dir']))
        trace.update(process_dlc(dlc_data))
    except:
        print("DLC file not found!")

    # Get reward info
    print("  4. Get Reward Status")
    reward_time, reward_label = get_reward_info(
        os.path.dirname(trace['save_dir'])
    )
    behav_reward = coordinate_event_behav(
        event_time=reward_time, 
        video_time=trace['video_time']
    )

    trace['reward_time'] = reward_time
    trace['reward_label'] = reward_label
    trace['behav_reward'] = behav_reward

    # Separate Trials
    print("  5. Separate Trials")
    onset, end, behav_freq_filtered, end_freq = identify_trials(
        dominant_freq=behav_freq,
        freq_thre=23
    )
    print(f"    Extracted {len(onset)} trials")

    trace.update({
        'onset_frames': onset,
        'end_frames': end,
        'dominant_freq_filtered': behav_freq_filtered,
        'end_freq': end_freq
    })
    return trace

def save_trace(trace: dict) -> None:
    print(f"    Finally, confirm {len(trace['onset_frames'])} trials")
    print("  6. Save Data -------------------------")
    
    with open(trace['save_dir'], 'wb') as handle:
        pickle.dump(trace, handle)

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print("Done. ----------------------------------", end='\n\n')


if __name__ == "__main__":
    from local_path import f1_behav

    run_all_sessions(f1_behav, 0)