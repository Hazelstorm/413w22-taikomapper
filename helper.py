import os
import torch
from torch.nn.functional import pad
import numpy as np
import json
import matplotlib.pyplot as plt
import hyper_param

"""
Converts <snap_num> to time in ms.
Parameters:
- bar_len: length of one bar (measure), in ms. Calculated as 60000/BPM.
- offset: the offset of snap #0 in ms, usually corresponding to the song's first beat.
- snap_val: number of snaps in one bar.

For example, if the song is at 200BPM and has offset 1000ms, bar_len is 300ms. 
If there are 4 snaps in one bar, snap #5 would occur at time 1375ms.
"""
def snap_to_ms(bar_len, offset, snap_num, snap_val=hyper_param.snap):
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
def get_num_snaps(bar_len, offset, ms, snap_val=hyper_param.snap):
    snap_num = (ms - offset) * snap_val / bar_len # inverse function of snap_to_ms
    snap_num = np.ceil(snap_num).astype(int)
    if snap_to_ms(bar_len, offset, snap_num) >= ms: # we may have predicted a value too high due to rounding error
        return snap_num
    return snap_num + 1
    
"""
Filters out ms from the model prediction and ground-truth matrices that don't fall on snaps.
bar_len and offset are used to calculate the times at which snaps occur.
Parameters:
- notes_data: note data, a numpy matrix of shape [N,5]
- unsnap_tolerance: number of ms of leniency. A note is considered to fall on a snap at s ms if it occurs in
  the time interval [s - unsnap_tolerance, s + unsnap_tolerance].
- numpy: Whether notes_data is a numpy array. If numpy=True, returns a Numpy array instead of a Torch tensor.
"""
def filter_to_snaps(notes_data, bar_len, offset, unsnap_tolerance, numpy=False):
    num_snaps = get_num_snaps(bar_len, offset, notes_data.shape[0])
    indices = snap_to_ms(bar_len, offset, np.arange(num_snaps))
    out = None
    if numpy:
        notes_data_padded = np.pad(notes_data, (unsnap_tolerance, unsnap_tolerance), constant_values=(0, 0))
    else:
        notes_data_padded = pad(notes_data, pad=(unsnap_tolerance, unsnap_tolerance))
    for i in range(0, 2 * unsnap_tolerance + 1):
        out_cur = notes_data_padded[indices + i]
        if out is None:
            out = out_cur
        else:
            if numpy:
                out = np.maximum(out, out_cur)
            else:
                out = torch.maximum(out, out_cur)
    return out

"""
(Approximately) converts <ms> to snap number, and rounds the resulting snap into an integer if close enough.
Returns None if <ms> is not close enough to a whole snap.

For example, if the song is at 200BPM and has offset 1000ms, and there are 4 snaps in one bar,
Whole snaps would occur at 1000ms, 1075ms, 1150ms, 1225ms, ...

Parameters:
- bar_len: length of one bar (measure), in ms. Calculated as 60000/BPM.
- offset: the offset of snap #0 in ms, usually corresponding to the song's first beat.
"""
def ms_to_snap(bar_len, offset, ms, snap_val=hyper_param.snap):
    snap_num = (ms - offset) / (bar_len / snap_val)
    error = abs(snap_num - round(snap_num))
    if error < 0.02 * hyper_param.snap:
        return round(snap_num)
    
"""
Convert indicies into one-hot vectors by
    1. Creating an identity matrix of shape [total, total]
    2. Indexing the appropriate columns of that identity matrix
"""
def make_onehot(indicies):
    I = torch.eye(hyper_param.snap)
    if torch.cuda.is_available():
        I = I.cuda()
    return I[indicies]

"""
Pulls from the spectrogram severals window of audio centered at each snapped ms of the audio with a window of <win_size> ms added on either side.
If the window extends past the start/end of the spectrogram, padding values will be inserted equal to the minimum value of the spectrogram.

Parameters:
- spectro: the spectrogram to pull audio from, a numpy matrix of shape [N, 40]
- bar_len: bar_len of the audio
- offset: offset of the audio
- win_size: how large the window should be on each side of the snapped ms
"""
def get_audio_around_snaps(spectro, bar_len, offset, win_size):
    win_length = win_size * 2 + 1
    
    padded_spectro = pad(spectro, pad=(0, 0, win_size, win_size), value=torch.min(spectro)) # pad the spectro on each side with minimum values
    num_snaps = get_num_snaps(bar_len, offset, spectro.shape[0])
    indices = snap_to_ms(bar_len, offset, np.arange(num_snaps))
    index_num = make_onehot(torch.arange(num_snaps) % hyper_param.snap)
    
    audio_windows = torch.zeros([num_snaps, 0, hyper_param.n_mels]) # [indices, win_length, 40]
    if torch.cuda.is_available():
        audio_windows = audio_windows.cuda()
        padded_spectro = padded_spectro.cuda()
    
    for i in range(win_length):
        audio_slices = padded_spectro[indices + i, :]
        audio_slices = torch.unsqueeze(audio_slices, 1)
        audio_windows = torch.cat([audio_windows, audio_slices], axis=1)
        
    audio_windows = torch.flatten(audio_windows, start_dim=1)
    audio_windows = torch.cat([audio_windows, index_num], axis=1)
        
    return audio_windows

"""
Returns numpy data (audio_data, notes_data, timing) stored by create_data() in datasets.py.
"""
def get_npy_data(path):
    notes_data = np.load(os.path.join(path, "notes_data.npy"))
    tail, head = os.path.split(path)
    audio_data = np.load(os.path.join(os.path.split(tail)[0], "audio", head, "audio_data.npy"))
    with open(os.path.join(path, "timing_data.json")) as file:
        timing_data = json.load(file)
    return audio_data, timing_data, notes_data

"""
Returns the total number of snaps with notes and the total number of snaps with no notes in a tuple.
"""
def get_note_ratio(notes_data): # useful for computing expected weight of loss function
    notes = torch.sum(notes_data > 0).item()
    total = notes_data.shape[1]
    no_notes = total - notes
    return notes, no_notes

"""
Returns the total number of finisher notes and the total number of non-finisher notes.
"""
def get_finisher_ratio(notes_data):
    finishers = torch.sum(notes_data >= 3).item()
    total = notes_data.shape[1]
    not_finishers = total - finishers
    return finishers, not_finishers