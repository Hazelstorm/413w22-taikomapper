import os
import numpy as np
import matplotlib.pyplot as plt
from hyper_param import *

SNAP = get_snap() # Number of snaps in one bar
WINDOW_SIZE = get_window_size() # Input time window (in ms) around each snap
MAX_SNAP = get_max_snap() # Maximum number of snaps in song

"""
Returns numpy data stored by create_data() in datasets.py. 

If get_timing is False (default), returns audio_data and notes_data only
Otherwise, returns audio_data, notes_data, timing
"""
def get_npy_data(path, get_timing=False):
    notes_data = np.load(os.path.join(path, "notes_data.npy"))
    tail, head = os.path.split(path)
    audio_data = np.load(os.path.join(os.path.split(tail)[0], "audio", head, "audio_data.npy"))
    if not get_timing:
        return audio_data, notes_data
    timing = np.load(os.path.join(path, "timing.npy"))
    return audio_data, notes_data, timing