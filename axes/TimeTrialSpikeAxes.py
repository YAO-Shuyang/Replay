import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import os

def TimeTrialSpikeAxes(
    ax: Axes,
    trial_start: np.ndarray,
    trial_end: np.ndarray,
    spikes: np.ndarray,
    time: np.ndarray,
    title: str = ""
    freq
) -> tuple[Axes, list]:
    if trial_start.shape[0] != trial_end.shape[0]:
        raise ValueError(
            f"Trial_start {trial_start.shape[0]} and trial_end "
            f"{trial_end.shape[0]} have different length."
        )
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Convert distint trials to distinct y values
    y_label = np.zeros(time.shape[0])
    for i in range(trial_start.shape[0]):
        indices = np.where(
            (time >= trial_start[i]) & (time <= trial_end[i])
        )[0]
        y_label[indices] = i
        time[indices] = time[indices] - trial_start[i]
    
    insert_nan_indices = np.where(np.ediff1d(y_label) != 0)[0]
    y_label = np.insert(y_label.astype(np.float64), insert_nan_indices+1, np.nan)
    time = np.insert(time.astype(np.float64), insert_nan_indices+1, np.nan)
    time += np.random.rand(time.shape[0]) - 0.5
    spikes = np.insert(spikes.astype(np.float64), insert_nan_indices+1, np.nan)
    
    a = ax.plot(time, y_label, color = 'grey', linewidth = 0.5)
    b = ax.plot(time[spikes == 1], spikes[spikes == 1], '|', color = 'red', markeredgewidth = 0. markersize = 4)
    c = ax.axvline(time, y_label, color = 'black', ls = '--', linewidth = 0.5)
    ax.set_xlim(np.min(time)-1, np.max(time)+1)
    ax.set_xticks([0, int(np.max(time)/2), int(np.max(time))])
    ax.set_xlabel('Time to Onsets (s)')
    ax.set_ylabel('Trial')
    ax.set_yticks([0, trial_start.shape[0]-1], [1, trial_start.shape[0]])
    ax.set_ylim(-0.5, trial_start.shape[0]-0.5)
    ax.set_title(title)
    
    return ax, a + b + c