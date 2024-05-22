import numpy as np
from replay.preprocess.read import read_behav
from datetime import datetime, timedelta

def transform_time_format(time_stamp: np.ndarray) -> np.ndarray:
    """
    transform date time, xx:xx:xx.xxx to xx.xxx

    Parameters
    ----------
    time_stamp : np.ndarray
        date time, xx:xx:xx.xxx

    Returns
    -------
    np.ndarray
        date time, xx.xxx

    Example
    --------
    >>> time = np.array(['12:34:56.789', '01:23:45.678', '23:59:59.999'])
    >>> time = transform_time_format(time)
    >>> time
    array([45296.789,  5025.678, 86399.999])
    """
    # Split the time strings by ':' and '.' to separate hours, minutes, seconds, and microseconds
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

def process_SMT_data(dir_name: str) -> np.ndarray:
    """
    process behavioral data
    
    Returns
    -------
    np.ndarray
        processed behavioral data, with a shape of 3 x T
        i.e., time, frequency, and behavioral states
    """
    # Read Frequency
    f_behav_freq = read_behav(dir_name=dir_name, sheet_name="Frequency")
    freq = np.array(f_behav_freq['Frequency'])
    freq_time = transform_time_format(np.array(f_behav_freq['Time']).astype('>U12'))
    print(freq.shape, freq_time.shape)
    print(freq_time)

    # Read 

if __name__ == "__main__":
    from replay.local_path import f1
    import doctest
    #doctest.testmod()

    process_SMT_data(dir_name=f1['behav_path'][0])

    # Test examples with
    