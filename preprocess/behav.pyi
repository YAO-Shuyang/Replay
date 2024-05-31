import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

# Get Lever Status
def read_dlc(
    dir_name: str, 
    find_char: str = '805000.csv',
    bodyparts: list[str] = [
        'FixedPoint1', 'FixedPoint2', 'Chin', 'Tube_top', 'Tube_bottom',
        'Lever_right', 'Lever_left'
    ],
    **kwargs
) -> dict: ...
def process_dlc(dlc_data: dict) -> dict: ...

# Get Reward Status and Lever states
def read_behav(
    dir_name: str, 
    file_name: Optional[str] = None,
    **kwargs
) -> pd.DataFrame: ...
def transform_time_format(time_stamp: np.ndarray) -> np.ndarray: ...
def get_reward_info(dir_name: str) -> np.ndarray: ...
def get_lever_event(dir_name: str) -> np.ndarray: ...
def coordinate_event_behav(
    event_time: np.ndarray, 
    video_time: np.ndarray
) -> np.ndarray: ...

# Identify trials' onset and end.
def identify_trials(
    dominant_freq: np.ndarray,
    freq_thre: int = 23
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...