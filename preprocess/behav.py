import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

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
    file_name: Optional[str] = None,
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
    if file_name is None:
        # find files whose name contain "READY.xlsx"
        files = [f for f in os.listdir(dir_name) if "READY.xlsx" in f]
        if len(files) == 0:
            raise FileNotFoundError(f"Fail to find READY.xlsx in {dir_name}")
        elif len(files) > 1:
            raise Exception(
                f"Found more than one READY.xlsx in {dir_name}:\n"
                f"{files}"
            )
        else:
            file_name = files[0]
 
    f_dir = os.path.join(dir_name, file_name)
    return pd.read_excel(f_dir, **kwargs)

def transform_time_format(time_stamp: np.ndarray) -> np.ndarray:
    """
    transform date time, xx:xx:xx.xxx to xx.xxx

    Parameters
    ----------
    time_stamp : np.ndarray
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

def identify_trials_SMT(
    lever_states: np.ndarray,
    dorm_frequency: np.ndarray
):
    """
    Identify trials for SMT task.
    """
    plt.plot(np.arange(dorm_frequency.shape[0]), dorm_frequency)
    plt.show()



if __name__ == "__main__":
    from replay.local_path import f1
    import doctest
    #doctest.testmod()

    #process_SMT_data(dir_name=f1['behav_path'][0])

    # Test examples with
    dlc_data = read_dlc(r"E:\behav\SMT\27049\20220516")
    process_dlc(dlc_data)

    print(get_reward_info(r"E:\behav\SMT\27049\20220516"))
    