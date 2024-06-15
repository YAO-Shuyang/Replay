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
from mazepy.basic._cshuffle import _shift_shuffle, _shift_shuffle_kilosort

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
    spike_time: np.ndarray,
    trial_start: np.ndarray,
    trial_end: np.ndarray,
    t_window: float
) -> np.ndarray:
    time_bin = np.full_like(spike_time, np.nan)
    
    for i in range(trial_start.shape[0]):
        idx = np.where(
            (spike_time >= trial_start[i]) & (spike_time <= trial_end[i])
        )[0]

        time_bin[idx] = (spike_time[idx] - trial_start[i]) // t_window

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
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    for n in range(1000, 16000, 1000):
        print(n, (np.argmin(np.abs(trace['frequencies'] - n)) - 23) // 2)
    # Read neuropixel data.
    print("  1. Read Neuropixel Data")
    trace.update(read_npxdata(f['npx_folder'][i]))

    # Align the time stamps of neuropixel and behavior data.
    print("  2. Align Time Stamps")
    npx_init_time = transform_time_format(f['npx_init_time'][i])[0]
    behav_init_time = transform_time_format(f['behav_init_time'][i])[0]
    behav_time = trace['video_time'] - npx_init_time + behav_init_time
    
    # Get the spikes within trials
    print("  3. Get Spikes Within Trials")
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

    print("  4. Frequency Domain Analysis:")
    print("     a. Compute tuning curve.")
    spike_train: SpikeTrain = KilosortSpikeTrain.get_spike_train(
        activity=spikes_remain,
        time=spike_time_remain * 1000, # Convert to ms
        variable=VariableBin(spike_freq)
    )

    trace['spike_train_within'] = spike_train.to_array().astype(np.int64)
    trace['spike_time_within'] = spike_time_remain.astype(np.float64)
    trace['spike_freqbin_within'] = spike_freq.astype(np.int64)
    
    # Calculate the tuning curve in the frequency domain.
    freq_rate_map = spike_train.calc_tuning_curve(
        nbins=76,
        kilosort_spikes=spikes_remain
    )
    gkernel = GaussianKernel1d(n = 76, sigma=0.8)
    smoothed_rate_map = freq_rate_map.smooth(gkernel)
    trace['freq_rate_map'] = smoothed_rate_map
    trace['freq_smoothed_map'] = smoothed_rate_map
    
    print("     b. Shuffle and test if neurons tune to frequency.")
    # Shuffle to test if neurons tune to frequency
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
    
    # Calculate tuning curve in the time domain
    print("  5. Time Domain Analysis:")
    print("     a. Compute tuning curve.")
    time_bin = get_time_bins_labels(
        spike_time=spike_time_remain,
        trial_start=behav_time[trace['onset_frames']],
        trial_end=behav_time[trace['end_frames']],
        t_window=0.1
    )
    
    assert np.where(np.isnan(time_bin))[0].shape[0] == 0
    time_bin = VariableBin(time_bin.astype(np.int64))
    nbins = np.max(time_bin) + 1
    
    time_spike_train = KilosortSpikeTrain.get_spike_train(
        activity=spikes_remain,
        time=spike_time_remain * 1000,
        variable=VariableBin(time_bin)
    )
    time_rate_map = time_spike_train.calc_tuning_curve(
        nbins=nbins,
        kilosort_spikes=spikes_remain
    )
    trace['time_rate_map'] = time_rate_map
    trace['time_smoothed_map'] = time_rate_map
    trace['spike_timebin_within'] = time_bin
    
    # Shuffle to test if neurons tune to time
    print("     b. Shuffle and test if neurons tune to time.")
    res = _shift_shuffle_kilosort(
        spikes = spikes_remain.astype(np.int64),
        dtime = time_spike_train.get_dt().astype(np.float64),
        variable = time_bin,
        nbins = nbins,
        n_shuffle = 1000,
        info_thre = 95
    )
    trace['is_timecell'] = res[:, 0].astype(np.int64)
    trace['TI'] = res[:, 1].astype(np.float64)
    trace['TI_shuf'] = res[:, 2].astype(np.float64)

    # Calculate the neural trajectory
    print("  6. Calculate the neural trajectory.")
    neural_traj = spike_train.calc_neural_trajectory(
        t_window=50,
        step_size=50
    )
    trace['neural_traj'] = neural_traj.to_array()
    trace['time_traj'] = neural_traj.time
    trace['variable_traj'] = neural_traj.variable
    
    print("  7. Save Data.")
    save_dir = os.path.join(os.path.dirname(f['Trace Behav File'][i]), 'trace.pkl')
    with open(save_dir, 'wb') as handle:
        pickle.dump(trace, handle)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print(f"    {save_dir} is saved. Done.------------------", end='\n\n')
    
    return f

if __name__ == '__main__':
    from replay.local_path import f1_behav

    for i in range(len(f1_behav)):
        if i != 0:
            continue
        f1_behav = process_neural_activity(f1_behav, i)
        
    