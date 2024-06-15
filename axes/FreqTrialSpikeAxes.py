import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import os

def FreqTrialSpikeAxes(
    ax: Axes,
    trial_start: np.ndarray,
    trial_end: np.ndarray,
    spikes: np.ndarray,
    freq: np.ndarray,
    freq_time: np.ndarray,
    title: str = ""
) -> tuple[Axes, list]:
    if trial_start.shape[0] != trial_end.shape[0]:
        raise ValueError(
            f"Trial_start {trial_start.shape[0]} and trial_end "
            f"{trial_end.shape[0]} have different length."
        )
    if freq.shape[0] != freq_time.shape[0]:
        raise ValueError(
            f"Freq {freq.shape[0]} and freq_time {freq_time.shape[0]}"
            f"have different length."
        )
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Convert distint trials to distinct y values
    y_label = np.zeros(freq_time.shape[0])
    for i in range(trial_start.shape[0]):
        indices = np.where(
            (freq_time >= trial_start[i]) & (freq_time <= trial_end[i])
        )[0]
        y_label[indices] = i
    
    insert_nan_indices = np.where(np.ediff1d(y_label) != 0)[0]
    y_label = np.insert(y_label.astype(np.float64), insert_nan_indices+1, np.nan)
    freq = np.insert(freq.astype(np.float64), insert_nan_indices+1, np.nan)
    freq += np.random.rand(freq.shape[0]) - 0.5
    spikes = np.insert(spikes.astype(np.float64), insert_nan_indices+1, np.nan)
    
    a = ax.plot(freq, y_label, color = 'grey', linewidth = 0.5)
    b = ax.plot(freq[spikes == 1], spikes[spikes == 1], '|', color = 'red', markeredgewidth = 0. markersize = 4)
    c = ax.axvline(freq, y_label, color = 'black', ls = '--', linewidth = 0.5)
    ax.set_xlim(np.min(freq)-1, np.max(freq)+1)
    ax.set_xticks([0, 6, 11, 17, 23, 29, 35, 40, 46, 52, 58, 64, 70, 75], [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Trial')
    ax.set_yticks([0, trial_start.shape[0]-1], [1, trial_start.shape[0]])
    ax.set_ylim(-0.5, trial_start.shape[0]-0.5)
    ax.set_title(title)
    
    return ax, a + b + c