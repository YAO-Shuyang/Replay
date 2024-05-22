import pandas as pd
import numpy as np
import os
import json
from typing import Optional

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

def read_brain_region(
    dir_name: str,
    n_neuron: int, 
    file_name: str = "brain_region.json"
) -> np.ndarray:
    """
    read recording position for each isolated neuron.
    
    Parameters
    ----------
    dir_name : str
        Directory of brain region data
    n_neuron : int
        Number of neurons
    file_name : str
        File name of brain region data

    Returns
    -------
    np.ndarray
        brain region data
    """
    f_dir = os.path.join(dir_name, file_name)
    if os.path.exists(f_dir) == False:
        raise FileNotFoundError(
            f"Fail to Load Brain Region: {f_dir} was not found!"
        )
    
    # Load JSON file
    with open(os.path.join(dir_name, file_name)) as f:
        data = json.load(f)

    # Test the uniqueness of indexes
    all_cell_index = np.concatenate([data[k] for k in data.keys()])
    if len(all_cell_index) != len(set(all_cell_index)):
        raise Exception(
            f"Cell indexes are not unique, namely there're neurons "
            f"being repeated!"
        )

    # Test potential overflow
    if np.max(all_cell_index) > n_neuron:
        raise OverflowError(
            f"Max cell indexes {np.max(all_cell_index)} should be less "
            f"than {n_neuron}."
        )
    
    brain_region = np.zeros(n_neuron, dtype='U6')
    for k in data.keys():
        brain_region[np.array(data[k])-1] = np.repeat(k, len(data[k]))

    return brain_region

def read_npy(
    dir_name: str,
    file_name: str = "spike_clusters.npy"
) -> np.ndarray:
    """
    read numpy NPY file
    """
    f_dir = os.path.join(dir_name, file_name)
    if os.path.exists(f_dir) == False:
        raise FileNotFoundError(
            f"Fail to Load Neural Activity: {f_dir} was not found!"
        )
    return np.load(f_dir, allow_pickle=True)

def read_neural_activity(
    dir_name: str,
    file_name: str = "spike_clusters.npy"
) -> np.ndarray:
    """
    read clustered neural activity processed by kilosort4
    """
    return read_npy(dir_name=dir_name, file_name=file_name)

def read_neural_time(
    dir_name: str,
    file_name: str = "spike_times.npy"
) -> np.ndarray:
    """
    read related neural time
    """
    return read_npy(dir_name=dir_name, file_name=file_name)

def read_good_units(
    dir_name: str,
    file_name: str = "final_goodunits.txt"
) -> np.ndarray:
    """
    Read units that are qualified as good units.
    """
    f_dir = os.path.join(dir_name, file_name)
    if os.path.exists(f_dir) == False:
        raise FileNotFoundError(
            f"Fail to Load Good Units: {f_dir} was not found!"
        )
    return np.loadtxt(f_dir).astype(np.int64)

def read_npxdata(dir_name: str) -> dict:
    """read all neuropixel data from the given directory"""
    spike_times = read_neural_time(f1['npx_path'][0])
    spike_clusters = read_neural_activity(f1['npx_path'][0])
    
    n_neuron = np.max(spike_clusters)
    brain_region = read_brain_region(f1['npx_path'][0], n_neuron)
    good_units = read_good_units(f1['npx_path'][0])

    # Test length consistency
    assert np.where(brain_region != '')[0].shape[0] == len(good_units)
    assert np.where(
        np.where(brain_region != '')[0] + 1 - good_units != 0
    )[0].shape[0] == 0

    return {
        'spike_times': spike_times,
        'spike_clusters': spike_clusters,
        'brain_region': brain_region,
        'good_units': good_units,
        'n_neuron': n_neuron
    }

if __name__ == "__main__":
    from replay.local_path import f1

    # Test neural time
    spike_times = read_neural_time(dir_name=f1['npx_path'][0])
    print(spike_times, type(spike_times))
    print(spike_times.shape, spike_times.dtype, end="\n\n")

    # Test neural activity
    spike_clusters = read_neural_activity(dir_name=f1['npx_path'][0])
    print(spike_clusters, type(spike_clusters))
    print(spike_clusters.shape, spike_clusters.dtype, end="\n\n")
    
    # Test brain region
    n_neuron = np.max(spike_clusters)
    brain_region = read_brain_region(
        dir_name=f1['npx_path'][0], 
        n_neuron=n_neuron
    )
    print(brain_region, type(brain_region), brain_region.shape)

    # Test good units
    good_units = read_good_units(dir_name=f1['npx_path'][0])
    print(good_units, type(good_units), good_units.shape)

    # Test length consistency
    assert np.where(brain_region != '')[0].shape[0] == len(good_units)
    assert np.where(np.where(brain_region != '')[0] + 1 - good_units != 0)[0].shape[0] == 0