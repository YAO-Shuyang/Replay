from replay.local_path import f1_behav
from replay.preprocess.frequency import read_audio, sliding_stft

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    for i in range(len(f1_behav)):
        if i != 3:
            continue
        dir_name = f1_behav['recording_folder'][i]
        audio = read_audio(dir_name)

        frequencies, magnitudes = sliding_stft(
            audio['audio'], 
            duration = audio['duration'],# 1800.19, 
            targ_frames = audio['video_frames'], #54005,
            n = 512
        )

        magnitudes = magnitudes / np.max(magnitudes, axis=0)
        
        ax = plt.axes()
        ax.imshow(magnitudes, cmap='hot', interpolation='nearest')
        ax.set_aspect('auto')
        plt.show()