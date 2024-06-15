from replay.local_path import f1_behav, f1
import os
import pandas as pd

for i in range(len(f1_behav)):
    f1_behav.loc[i, "recording_folder"] = os.path.join(
        r"E:\behav",
        f1_behav['paradigm'][i],
        str(int(f1_behav['MiceID'][i])),
        str(int(f1_behav['date'][i])),
        'session ' + str(int(f1_behav['session'][i]))
    )
    
    f1_behav.loc[i, "Trace Behav File"] = os.path.join(
        r"E:\behav",
        f1_behav['paradigm'][i],
        str(int(f1_behav['MiceID'][i])),
        str(int(f1_behav['date'][i])),
        'session ' + str(int(f1_behav['session'][i])),
        'trace_behav.pkl'
    )
    
f1_behav.to_excel(r"E:\behav\behavioral_paradigms_behavoutput.xlsx", index=False)

for i in range(len(f1)):
    f1.loc[i, 'Trace File'] = os.path.join(
        r"E:\behav",
        f1['paradigm'][i],
        str(int(f1['MiceID'][i])),
        str(int(f1['date'][i])),
        'trace.pkl'
    )
    
f1.to_excel(r"E:\behav\behavioral_paradigms_neuractoutput.xlsx", index=False)
"""
import pickle
for i in range(len(f1_behav)):
    if os.path.exists(f1_behav['Trace Behav File'][i]):
        with open(f1_behav['Trace Behav File'][i], 'rb') as handle:
            trace = pickle.load(handle)
            
        trace['save_dir'] = f1_behav['Trace Behav File'][i]
        
        with open(f1_behav['Trace Behav File'][i], 'wb') as handle:
            pickle.dump(trace, handle)
"""