import numpy as np
import os
import pandas as pd
import pickle
import time
import copy as cp

from mazepy.datastruc.variables import KilosortSpikeTrain, VariableBin
from replay.preprocess.read import read_npxdata 
from replay.preprocess.behav import transform_time_format

"""
Process neural activity
"""

def get_within_trial_spike_index(
    spike_time: np.ndarray,
    trial_start: np.ndarray,
    trial_end: np.ndarray
) -> np.ndarray:
    """
    Get the spike index within each trial.

    Parameters
    ----------
    spike_time : np.ndarray (n_spikes, )
        The time stamps of each spikes from all identified units.
    trial_start : np.ndarray
        The start time of each trial. (n_trials, )
    trial_end : np.ndarray
        The end time of each trial. (n_trials, )
    """
    assert len(trial_start) == len(trial_end)

    return np.concatenate([np.where(
        (spike_time >= trial_start[i]) & (spike_time <= trial_end[i])
    )[0] for i in range(len(trial_start))])

def get_spike_related_freq(
    spike_time: np.ndarray,
    freq_time: np.ndarray,
    onsets: np.ndarray,
    ends: np.ndarray,
    freq: np.ndarray
) -> np.ndarray:
    """
    get the frequency related to each spike.

    Parameters
    ----------
    spike_time : np.ndarray
        The time stamps of each spikes from all identified units.
    freq_time : np.ndarray
        The time stamps of the frequency.
    freq : np.ndarray
        The frequency of each behavioral frame.
    """
    spike_freq = np.array([])

    for i in range(len(onsets)):
        spike_time_temp = spike_time[np.where(
            (spike_time >= freq_time[onsets[i]]) & (spike_time <= freq_time[ends[i]])
        )[0]]
        arr = np.repeat(
            freq_time[np.newaxis, onsets[i]:ends[i]+1], 
            len(spike_time_temp), 
            axis = 0
        )
        diff = np.abs(arr - spike_time_temp[:, np.newaxis])
        spike_freq = np.concatenate([spike_freq,  freq[onsets[i]:ends[i]+1][np.argmin(diff, axis = 1)]])

    return spike_freq

def process_neural_activity(
    f: pd.DataFrame,
    i: int
) -> None:
    """
    Main function to process neural activity and integrate with behavior data.

    Parameters
    ----------
    f : pd.DataFrame
        Sheet contains information of all sessions.
    i : int
        Trial number
    """

    # Read behavior data
    with open(f['Trace Behav File'][i], 'rb') as handle:
        trace = pickle.load(handle)
    
    print(
        f"{i}  Mouse {trace['mouse']} {trace['date']} {trace['paradigm']}"
        f"-----------------------"
    )

    # Further process behavioral data.
    npx_init_time = transform_time_format(f['npx_init_time'][i])[0]
    behav_init_time = transform_time_format(f['behav_init_time'][i])[0]
    behav_time = trace['video_time'] - npx_init_time + behav_init_time
    
    within_trial_spike_index = get_within_trial_spike_index(
        trace['spike_times'],
        behav_time[trace['onset_frames']],
        behav_time[trace['end_frames']]
    )

    spikes_remain = trace['spike_clusters'][within_trial_spike_index]
    spike_time_remain = trace['spike_times'][within_trial_spike_index]
    frequency = trace['dominant_freq_filtered'][trace['dominant_freq_filtered'] > 0]

    from replay.preprocess.neuact import get_spike_related_freq

    spike_freq = get_spike_related_freq(
        spike_time_remain, 
        behav_time,
        trace['onset_frames'],
        trace['end_frames'],
        trace['dominant_freq_filtered']
    ) - 23 # set to 0

    assert spike_time_remain.shape[0] == spike_freq.shape[0]

    neural_activity = KilosortSpikeTrain(
        activity=spikes_remain,
        time_stamp=spike_time_remain,
        variable=VariableBin(spike_freq)
    )

    trace['Spikes'] = cp.deepcopy(neural_activity.activity)
    
    
    

if __name__ == '__main__':
    from replay.local_path import f1
