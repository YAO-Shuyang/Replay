import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Union
import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from scipy.fft import rfftfreq

from tqdm import tqdm

"""
Author: @Shuyang Yao

Paradigms:
Aronov et al., 2017 {https://www.nature.com/articles/nature21692}

SMT: Sound Modulation Task
PP: Passive Playback
PPR: Pasive Playback with Rewards
"""

def read_dlc(
    dir_name: str, 
    find_char: str = '805000.csv',
    bodyparts: list[str] = [
        'FixedPoint1', 'FixedPoint2', 'Chin', 'Tube_top', 'Tube_bottom',
        'Lever_right', 'Lever_left'
    ],
    **kwargs
) -> dict:
    """read Deeplabcut-processed data
    
    Parameters
    ----------
    dir_name : str
        Directory of behavior data
    find_char : str
        File name of behavior data
    bodyparts : list[str]
        List of bodyparts
    kwargs : dict
        Additional arguments for pd.read_csv

    Returns
    -------
    dict
        dlc data
    """
    # find files whose name contain find_char
    files = [f for f in os.listdir(dir_name) if find_char in f]
    if len(files) == 0:
        raise FileNotFoundError(f"Fail to find {find_char} in {dir_name}")
    elif len(files) > 1:
        raise Exception(
            f"Found more than one {find_char} in {dir_name}:\n"
            f"{files}"
        )
    else:
        file_name = files[0]

    f_dir = os.path.join(dir_name, file_name)
    f = pd.read_csv(f_dir, header=[1,2], **kwargs)
    
    Data = {}
    for bodypart in bodyparts:
        coord = np.zeros((len(f), 2), dtype = np.float64)
        coord[:, 0] = f[bodypart, 'x']
        coord[:, 1] = f[bodypart, 'y']
        Data[bodypart] = coord

    return Data

def process_dlc(dlc_data: dict) -> dict:
    """
    Get the the status of the lever, whether consuming the water reward.
    """
    assert 'Lever_right' in dlc_data.keys()
    assert 'Lever_left' in dlc_data.keys()
    
    lever_depth = dlc_data['Lever_right'][:, 1] - dlc_data['Lever_left'][:, 1]
    thre = threshold_otsu(lever_depth)
    is_press = np.where(lever_depth >= thre, 0, 1)

    return {
        'lever_depth': lever_depth,
        'is_press': is_press,
        'lever_thre': thre
    }

def read_behav(
    dir_name: str, 
    file_name: str = "READY.xlsx",
    **kwargs
) -> pd.DataFrame:
    """
    read behavior excel sheet

    Parameters
    ----------
    dir_name : str
        Directory of behavior data
    file_name : str
        File name of behavior data
    kwargs : dict
        Additional arguments for pd.read_excel

    Returns
    -------
    pd.DataFrame
        behavior data
    """
    f_dir = os.path.join(dir_name, file_name)
    if not os.path.exists(f_dir):
        raise FileNotFoundError(f"Fail to find {f_dir}")
    return pd.read_excel(f_dir, **kwargs)

def transform_time_format(time_stamp: Union[np.ndarray, float]) -> np.ndarray:
    """
    transform date time, xx:xx:xx.xxx to xx.xxx

    Parameters
    ----------
    time_stamp : np.ndarray or float
        date time, xx:xx:xx.xxx

    Returns
    -------
    converted time stamp : np.ndarray
        date time, xx.xxx

    Example
    --------
    >>> time = np.array(['12:34:56.789', '01:23:45.678', '23:59:59.999'])
    >>> converted_time = transform_time_format(time)
    >>> converted_time
    array([45296.789,  5025.678, 86399.999])
    """
    # Split the time strings by ':' and '.' to separate hours, 
    # minutes, seconds, and microseconds
    if isinstance(time_stamp, str):
        time_stamp = np.array([time_stamp])

    converted_time = np.zeros_like(time_stamp, dtype = np.float64)

    for i in range(len(time_stamp)):
        time_obj = datetime.strptime(time_stamp[i], "%H:%M:%S.%f").time()
        time_delta = timedelta(
            hours=time_obj.hour, 
            minutes=time_obj.minute, 
            seconds=time_obj.second, 
            microseconds=time_obj.microsecond
        )
        converted_time[i] = time_delta.total_seconds()

    return converted_time.astype(np.float64)

def find_freq_bin(
    freq: Union[float, np.ndarray],
    frequencies: np.ndarray
) -> Union[int, np.ndarray]:
    """
    Get the bin ID for input frequency

    Parameters
    ----------
    freq : float | np.ndarray
        The frequency or frequencies to be converted.
    frequencies : np.ndarray
        The frequency centers for each freq bin.

    Returns
    -------
    int | np.ndarray
        The bin ID(s)
    """
    if isinstance(freq, float):
        df = np.abs(frequencies - freq)
        return np.argmin(df)
    else:
        freqs = np.repeat(frequencies[np.newaxis, :], freq.shape[0], axis = 0)
        df = np.abs(freqs - freq[:, np.newaxis])
        return np.argmin(df, axis = 1)

def get_freqseq_info(
    dir_name: str, 
    frequencies: Optional[np.ndarray] = None, 
    fps: int = 44100,
    n_fft: int = 512
) -> np.ndarray:
    """Get the last freq time stamp and freq."""
    f = read_behav(dir_name=dir_name, sheet_name = "Frequency")
    events = np.where(f['事件'] == "声音播放", 0, 1)
    transitions = np.append(np.ediff1d(events), -1)

    final_freq_idx = np.where(transitions == -1)[0]
    final_freq = np.array(f['频率'][final_freq_idx], dtype=np.float64)

    freq_time = transform_time_format(np.array(f['测试时间'])[final_freq_idx])

    if frequencies is None:
        frequencies = rfftfreq(n = n_fft, d = 1 / fps)

    final_freq = find_freq_bin(final_freq, frequencies = frequencies)

    return freq_time, final_freq

def get_reward_info(dir_name: str) -> np.ndarray:
    """
    Get and process reward information
    
    Returns
    -------
    reward_time : np.ndarray
    reward_label : np.ndarray
        1 - 液体奖励
        2 - 液体输出
        3 - 饮水
    """
    # Read Events
    f = read_behav(dir_name=dir_name, sheet_name="Protocol Result Data")
    reward_events_idx = np.where(
        (f['事件'] == "液体奖励") |
        (f['事件'] == "液体输出") |
        (f['事件'] == "饮水")
    )[0]

    reward_events = f['事件'][reward_events_idx]
    reward_label = np.zeros(reward_events_idx.shape[0], dtype = np.int64)
    reward_label[np.where(reward_events == "液体奖励")[0]] = 1
    reward_label[np.where(reward_events == "液体输出")[0]] = 2
    reward_label[np.where(reward_events == "饮水")[0]] = 3

    reward_time = transform_time_format(np.array(f['测试时间']))[reward_events_idx]
    return reward_time, reward_label

def get_lever_event(dir_name: str) -> np.ndarray:
    """
    Get the time when the lever is being released
    """
    f = read_behav(dir_name=dir_name, sheet_name="Protocol Result Data")
    releasing_idx = np.where(f['事件'] == "结束压杆")[0]
    return transform_time_format(np.array(f['测试时间']))[releasing_idx]

def coordinate_event_behav(
    event_time: np.ndarray, 
    video_time: np.ndarray
) -> np.ndarray:
    """
    Events reported by experimental systems are aligned with video time stamps
    here.
    """
    is_events = np.zeros(len(video_time), dtype = np.int64)

    arr = np.repeat(video_time[np.newaxis, :], event_time.shape[0], axis = 0)
    diff = np.abs(arr - event_time[:, np.newaxis])
    frames = np.argmin(diff, axis = 1)
    is_events[frames] = 1

    return is_events

def _search_trial_ends(
    onset_frame: int,
    dominant_freq: np.ndarray,
    freq_thre: int = 23
) -> tuple[int, np.ndarray]:
    """
    If frequency resets to
    """
    # Lever state transition
    
    for i in range(onset_frame, dominant_freq.shape[0]-1):
        if dominant_freq[i+1] < dominant_freq[i]:
            if dominant_freq[i+2] == dominant_freq[i] or dominant_freq[i+3] == dominant_freq[i]:
                dominant_freq[i+1] = dominant_freq[i]

            if dominant_freq[i+2] > dominant_freq[i] and dominant_freq[i+2] < dominant_freq[i]*1.4:
                dominant_freq[i+1] = dominant_freq[i]

        if dominant_freq[i+1] > 1.4 * dominant_freq[i]:
            if dominant_freq[i+2] == dominant_freq[i] or dominant_freq[i+3] == dominant_freq[i]:
                dominant_freq[i+1] = dominant_freq[i]

            if dominant_freq[i+2] > dominant_freq[i] and dominant_freq[i+2] < dominant_freq[i]*1.4:
                dominant_freq[i+1] = dominant_freq[i]

                
        if dominant_freq[i+1] < max(dominant_freq[i]- 2, freq_thre):
            if (dominant_freq[i+1] not in dominant_freq[onset_frame:i+1] or
                dominant_freq[i+1] == freq_thre):
                return i, dominant_freq
        
        if dominant_freq[i+1] > 1.4 * dominant_freq[i]:
            return i, dominant_freq

    plt.figure()
    plt.plot(np.arange(onset_frame, dominant_freq.shape[0]), dominant_freq[onset_frame:])
    plt.title("Failure in identifying the trial ends")
    plt.xlabel("Frame")
    plt.show()
    raise Exception(f"Fail to find trial end, onset frame: {onset_frame}")


def _merge_trials(
    onset_frame: int,
    end_frame: int,
    dominant_freq: np.ndarray
) -> np.ndarray:
    for i in range(onset_frame, end_frame):
        if dominant_freq[i+1] < dominant_freq[i]:
            dominant_freq[i+1] = dominant_freq[i]
        elif dominant_freq[i+1] > 1.4 * dominant_freq[i]:
            dominant_freq[i+1] = dominant_freq[i]
    
    return dominant_freq

def identify_trials(
    dominant_freq: np.ndarray,
    freq_thre: int = 23
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Identify trials for SMT task.

    Parameters
    ----------
    dominant_freq : np.ndarray
        The real-time dominant frequency, represented as freq bin ID.
        (0 to 256 representing 0Hz to 22050Hz)
    freq_thre : int
        Threshold to identify the onset of a trial (default: 23).
        2 kHz belongs to freq bin 23.

    Returns
    -------
    onset_frames : np.ndarray
        The onset frames of each trial.
    end_frames : np.ndarray
        The end frames of each trial.
    dominant_freq_filtered : np.ndarray
        The filtered dominant frequency.
    end_freq : np.ndarray
        The end frequency of each trial.
    """
    onset_frames = []
    end_frames = []
    end_freqs = []
    
    # Get onset frames 
    # Onset was defined as three successive frames with dominant frequency
    # equaling to threshold, i.e. 2 kHz
    dfreq = np.append(np.ediff1d(dominant_freq), -1)
    onset_idx = np.where((dominant_freq >= freq_thre) & (dfreq == 0))[0]


    gap_idx = 0
    for i in tqdm(onset_idx):
        if i < gap_idx:
            continue

        if dominant_freq[i] == freq_thre and i+1 not in onset_idx:
            continue

        # Consecutive 2 frames equal to 2 kHz
        onset_frame = i
        end_frame, dominant_freq = _search_trial_ends(
            onset_frame = i, 
            dominant_freq = dominant_freq,
        )

        if (dominant_freq[end_frame] < dominant_freq[end_frame-1] -2 or
            dominant_freq[end_frame] > dominant_freq[end_frame-1] + 2):
            end_frame -= 1

        gap_idx = end_frame + 1

        curr_freq = dominant_freq[onset_frame]
        end_freq = dominant_freq[end_frame]
        prev_freq = end_freqs[-1] if len(end_freqs) > 0 else np.nan
            
        # Check if this trial should be merged with the prior one.
        if len(end_freqs) == 0:
            if curr_freq == freq_thre and end_freq != freq_thre:
                onset_frames.append(onset_frame)
                end_frames.append(end_frame)
                end_freqs.append(end_freq)
        else:
            BEYOND_INTERVAL = end_frame - end_frames[-1] > 5
            SHOULD_MERGE = (
                (curr_freq >= prev_freq and curr_freq < 1.4 * prev_freq) or
                (curr_freq < prev_freq and 
                 curr_freq in dominant_freq[onset_frames[-1]:end_frames[-1]+1] and
                 curr_freq != freq_thre)
            )
                            
            if BEYOND_INTERVAL or not SHOULD_MERGE:
                if dominant_freq[onset_frame] == freq_thre and end_freq != freq_thre:
                    onset_frames.append(onset_frame)
                    end_frames.append(end_frame)
                    end_freqs.append(dominant_freq[end_frame])
            elif not BEYOND_INTERVAL and SHOULD_MERGE:
                dominant_freq = _merge_trials(
                    onset_frame = onset_frames[-1],
                    end_frame = end_frame,
                    dominant_freq = dominant_freq
                )
                end_frames.pop()
                end_freqs.pop()
                end_frames.append(end_frame)
                end_freqs.append(dominant_freq[end_frame])
            
    dominant_freq_filtered = np.zeros_like(dominant_freq)
    for i in range(len(onset_frames)):
        onset, end = onset_frames[i], end_frames[i]
        dominant_freq_filtered[onset:end+1] = dominant_freq[onset:end+1]

    onset_frames = np.array(onset_frames, dtype = np.int64)
    end_frames = np.array(end_frames, dtype = np.int64)
    end_freqs = np.array(end_freqs, dtype = np.int64)

    assert len(onset_frames) == len(end_frames) == len(end_freqs)
    
    return onset_frames, end_frames, dominant_freq_filtered, end_freqs



if __name__ == "__main__":
    from replay.local_path import f1
    import doctest
    #doctest.testmod()

    #process_SMT_data(dir_name=f1['behav_path'][0])

    # Test examples with
    dlc_data = read_dlc(r"E:\behav\SMT\27049\20220516\session 1")
    process_dlc(dlc_data)

    print(get_reward_info(r"E:\behav\SMT\27049\20220516\session 1"))
    