import os
import torch
from torch.nn.functional import pad
import numpy as np
import json
import matplotlib.pyplot as plt
from hyper_param import *

SNAP = get_snap() # Number of snaps in one bar
WINDOW_SIZE = get_window_size() # Input time window (in ms) around each snap
MAX_SNAP = get_max_snap() # Maximum number of snaps in song

"""
Converts <snap_num> to time in ms.
Parameters:
- bar_len: length of one bar (measure), in ms. Calculated as 60000/BPM.
- offset: the offset of snap #0 in ms, usually corresponding to the song's first beat.
- snap_val: number of snaps in one bar.

For example, if the song is at 200BPM and has offset 1000ms, bar_len is 300ms. 
If there are 4 snaps in one bar, snap #5 would occur at time 1375ms.
"""
def snap_to_ms(bar_len, offset, snap_num, snap_val=SNAP):
    ms = offset + (snap_num * bar_len / snap_val)
    return np.floor(ms).astype(int)

"""
Determines the number of snaps in a given song.
Parameters:
- bar_len: length of one bar (measure), in ms. Calculated as 60000/BPM.
- offset: the offset of snap #0 in ms, usually corresponding to the song's first beat.
- ms: the total length of the song in ms.
- snap_val: number of snaps in one bar.
"""
def get_num_snaps(bar_len, offset, ms, snap_val=SNAP):
    snap_num = (ms - offset) * snap_val / bar_len # inverse function of snap_to_ms
    snap_num = np.ceil(snap_num).astype(int)
    if snap_to_ms(bar_len, offset, snap_num) >= ms: # we may have predicted a value too high due to rounding error
        return snap_num
    return snap_num + 1 
    
"""
Filters out ms from the model prediction and ground-truth matrices that don't fall on snaps.
Parameters:
- model_out: the output of the model, a torch matrix of shape
"""
def filter_model_output(model_out, notes_data, timing_data):
    bar_len = timing_data["bar_len"].item()
    offset = timing_data["offset"].item()
    num_snaps = get_num_snaps(bar_len, offset, model_out.shape[0])
    indices = snap_to_ms(bar_len, offset, np.arange(num_snaps))
    y = model_out[indices]
    t = None
    unsnap_tolerance = 2
    notes_data_padded = pad(notes_data, pad=(0, 0, unsnap_tolerance, unsnap_tolerance))
    for i in range(0, 2 * unsnap_tolerance + 1):
        t_cur = notes_data_padded[indices + i]
        if t is None:
            t = t_cur
        else:
            t = torch.maximum(t, t_cur)
    return y,t

"""
(Approximately) converts <ms> to snap number, and rounds the resulting snap into an integer if close enough.
Returns None if <ms> is not close enough to a whole snap.

For example, if the song is at 200BPM and has offset 1000ms, and there are 4 snaps in one bar,
Whole snaps would occur at 1000ms, 1075ms, 1150ms, 1225ms, ...

Parameters:
- bar_len: length of one bar (measure), in ms. Calculated as 60000/BPM.
- offset: the offset of snap #0 in ms, usually corresponding to the song's first beat.
"""
def ms_to_snap(bar_len, offset, ms, snap_val=SNAP):
    snap_num = (ms - offset) / (bar_len / snap_val)
    error = abs(snap_num - round(snap_num))
    if error < 0.02 * SNAP:
        return round(snap_num)

"""
Returns numpy data stored by create_data() in datasets.py. 

If get_timing is False (default), returns audio_data and notes_data only
Otherwise, returns audio_data, notes_data, timing
"""
def get_npy_data(path):
    notes_data = np.load(os.path.join(path, "notes_data.npy"))
    tail, head = os.path.split(path)
    audio_data = np.load(os.path.join(os.path.split(tail)[0], "audio", head, "audio_data.npy"))
    with open(os.path.join(path, "timing_data.json")) as file:
        timing_data = json.load(file)
    return audio_data, timing_data, notes_data
