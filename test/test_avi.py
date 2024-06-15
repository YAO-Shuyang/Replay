import numpy as np
from replay.preprocess.frequency import read_audio, sliding_stft
from replay.preprocess.frequency import get_dominant_frequency
from matplotlib import pyplot as plt

#audio = read_audio(r"E:\LRJ\ori\SMT#27049\behavior\220517", file_name="220517.49.smt.mec.02-0517114357.avi")
audio = read_audio(r"E:\LRJ\processed\PPR#27124\behavior_ppr\0316", file_name="20230316_105917.mp4")
frequencies, magnitudes = sliding_stft(
    audio['audio'], 
    duration = audio['duration'],
    targ_frames = audio['video_frames'],
    n = 512
)
magnitudes = magnitudes / np.max(magnitudes, axis=0)
dorminant_freq = get_dominant_frequency(magnitudes)
plt.figure()
ax = plt.axes()
plt.imshow(magnitudes, cmap='hot', interpolation='nearest')
ax.set_aspect('auto')
plt.show()