import numpy as np
import os
import pandas as pd
import pickle
import time
import copy as cp

from mazepy.datastruc.variables import VariableBin
from mazepy.datastruc.neuact import KilosortSpikeTrain, SpikeTrain
from mazepy.datastruc.neuact import TuningCurve, NeuralTrajectory
from mazepy.datastruc.kernel import GaussianKernel1d
from replay.preprocess.read import read_npxdata 
from replay.preprocess.behav import transform_time_format
from mazepy.basic._cshuffle import _shift_shuffle

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

def get_time_bins_labels(
    time_traj: np.ndarray,
    trial_start: np.ndarray,
    trial_end: np.ndarray
):
    time_bin = np.full_like(time_traj, np.nan)
    
    for i in range(trial_start.shape[0]):
        idx = np.where(
            (time_traj >= trial_start[i]) & (time_traj <= trial_end[i])
        )[0]

        time_bin[idx] = np.arange(idx.shape[0])

    return time_bin

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
    
    # Read neuropixel data.
    trace.update(read_npxdata(f['npxdata'][i]))

    # Align the time stamps of neuropixel and behavior data.
    npx_init_time = transform_time_format(f['npx_init_time'][i])[0]
    behav_init_time = transform_time_format(f['behav_init_time'][i])[0]
    behav_time = trace['video_time'] - npx_init_time + behav_init_time
    
    # Get the spikes within trials
    within_trial_spike_index = get_within_trial_spike_index(
        trace['spike_times'],
        behav_time[trace['onset_frames']],
        behav_time[trace['end_frames']]
    )

    # Get the within-trial spikes and related times
    spikes_remain = trace['spike_clusters'][within_trial_spike_index]
    spike_time_remain = trace['spike_times'][within_trial_spike_index]

    from mazepy.basic.convert import coordinate_recording_time

    # Get the frequency related to each spike
    idx = coordinate_recording_time(
        spike_time_remain, 
        behav_time
    )
    # Bin 23 coorresponds to 2000Hz. Set the first frequency bin as 0.
    spike_freq = trace['dominant_freq_filtered'][idx] - 23
    # Ensure no negative frequency
    assert np.where(spike_freq < 0)[0].shape[0] == 0
    # According to the frequency bin number we used in identifying the spectrum
    # of audio (n_FFT = 512), there're 257 frequency bins uniformly ranging from
    # 0 Hz to 22050 Hz. The 2000 Hz and 15000 Hz correspond to Bin 23 and 174 
    # respectively. 
    
    # Reduce total bin number to 76. Each is equivalent to 172.2 Hz.
    spike_freq = spike_freq // 2

    assert spike_time_remain.shape[0] == spike_freq.shape[0]

    spike_train: SpikeTrain = KilosortSpikeTrain.get_spike_train(
        activity=spikes_remain,
        input_time=spike_time_remain * 1000, # Convert to ms
        variable=VariableBin(spike_freq)
    )

    trace['spike_train'] = spike_train.to_array()
    
    # Calculate the tuning curve.
    freq_rate_map = spike_train.calc_tuning_curve(
        nbins=76,
        kilosort_spikes=spikes_remain,
        kilosort_variables=VariableBin(spike_freq)
    )
    gkernel = GaussianKernel1d(n = 76, sigma=0.8)
    smoothed_rate_map = freq_rate_map.smooth(gkernel)
    trace['freq_rate_map'] = smoothed_rate_map
    trace['smoothed_rate_map'] = smoothed_rate_map
    
    res = _shift_shuffle(
        spikes = spike_train.astype(np.int64),
        dtime = spike_train.get_dt().astype(np.float64),
        variable = spike_freq.astype(np.int64),
        nbins = 76,
        n_shuffle = 1000,
        info_thre = 95
    )
    trace['is_freqcell'] = res[:, 0].astype(np.int64)
    trace['FI'] = res[:, 1].astype(np.float64)
    trace['FI_shuf'] = res[:, 2].astype(np.float64)
        
    neural_traj = spike_train.calc_neural_trajectory(
        t_window=50,
        step_size=50
    )
    
    time_bin = get_time_bins_labels(
        time_traj=neural_traj.time,
        trial_start=behav_time[trace['onset_frames']],
        trial_end=behav_time[trace['end_frames']]
    )
    
    
    trace['neural_trajectory'] = neural_traj
    trace['time_bin'] = time_bin


if __name__ == '__main__':
    from replay.local_path import f1_behav
