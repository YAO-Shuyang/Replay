"""
Apply PCA to the frequency magnitudes to analyze behavioral states.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def freq_pca(
    magnitudes: np.ndarray,
    n_components: int,
    **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies PCA to reduce the dimensionality of the magnitude matrix from an 
    STFT.

    Parameters
    ----------
    magnitudes : np.ndarray
        The magnitude matrix from an STFT
    n_components : int
        The number of components to keep
    **kwargs : dict
        Additional arguments for PCA

    Returns
    -------
    pca_results : np.ndarray
        The transformed data with shape (n_time_frames, n_components)
    explained_variance_ratio : np.ndarray
        The percentage of variance explained by each of the selected components
    """
    # Normalize the magnitude matrix
    #scaler = StandardScaler()
    #scaled_magnitudes = scaler.fit_transform(magnitudes.T)

    # Apply PCA
    pca = PCA(n_components=n_components, **kwargs)
    pca_results = pca.fit_transform(magnitudes.T)

    print(pca_results.shape)

    # Calculate explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    return pca_results, explained_variance_ratio

def visualize_pca_freq(pca_results: np.ndarray):
    """
    Visualize the PCA results.

    Parameters
    ----------
    pca_results : np.ndarray
        The PCA results
    """
    plt.plot(pca_results[:, 1], pca_results[:, 2], 'o', markeredgewidth = 0, markersize = 5)
    plt.xlabel('Component')
    plt.show()

if __name__ == "__main__":
    import pickle
    from replay.preprocess.frequency import sliding_stft, read_audio

    dir_name = r"E:\behav\SMT\27049\20220516\session 1"
    
    audio = read_audio(dir_name)

    frequencies, magnitudes = sliding_stft(
        audio['audio'], 
        duration = audio['duration'],# 1800.19, 
        targ_frames = audio['video_frames'], #54005,
        n = 512
    )

    pca_results, explained_variance_ratio = freq_pca(magnitudes, 10)

    visualize_pca_freq(pca_results)