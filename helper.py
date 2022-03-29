import sys, librosa, math, os, torch
import numpy as np
import matplotlib.pyplot as plt
from hyper_param import *

from pydub import AudioSegment # to convert mp3 to wav

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
    return math.floor(ms)

"""
Converts <ms> to snap number, and rounds the resulting snap into an integer if close enough.
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
Convert indicies into one-hot vectors by
    1. Creating an identity matrix of shape [total, total]
    2. Indexing the appropriate columns of that identity matrix
Shamelessly stolen from A1 ¯\_(ツ)_/¯
"""
def make_onehot(indicies, total=5): 
    I = np.eye(total)
    return I[indicies]

"""
Return whether bit i is set in x.
"""
def bit_flag(x, i):
    return bool(x & (1 << i))


"""
Converts an integer <note>, representing the note type in .osu files (see below), to
an integer representing the note type in our model.

In .osu files, note types are stored in a 4-bit integer as follows:
don: 0001 (1)
kat: 0011 or 1001 or 1011 (3 or 9 or 11)
don finisher: 0101 (5)
kat finisher: 0111 or 1101 or 1111 (7 or 13 or 15)

In our model, note types are stored as an integer between 1 and 4, as follows:
don: 1
kat: 2
don finisher: 3
kat finisher: 4
"""
def get_note_type(note):
    is_kat = bit_flag(int(note), 1) or bit_flag(int(note), 3)
    is_finish = bit_flag(int(note), 2)
    return 1 + is_kat + 2 * is_finish

"""
Return the number of snaps in the spectrogram spectro. 
"""
def get_num_snaps(spectro, bar_len, offset):
    return round((spectro.shape[0] - offset) // bar_len * SNAP)

"""
Given a list of <notes> from get_map_notes(), return a sequence of
<num_snaps> notes that list the note occurring at each snap (including no note).
"""
def get_note_data(notes, num_snaps):
    data = np.zeros((num_snaps, 5))
    for note in notes:
        data[note[0], note[1]] = 1
        
    # If no note on tick, set Empty flag to true
    data[:,0] = data[:,0] + 1 - np.sum(data, axis=1)
    
    return data


"""
Given the <filepath> to a .mp3 and a window size, returns a numpy array
containing spectrogram data for the .mp3 file.

The mp3 is sampled in a window of size 2 * WINDOW_SIZE around each snap.

convert=True converts mp3 files into wav before processing, as librosa doesn't
handle mp3 efficiently.
"""
def get_map_audio(filepath, convert=True, kwargs=get_mel_param()): 
    sr = kwargs['sr']
    try: 
        if convert and filepath.endswith(".mp3"):
            wav = AudioSegment.from_mp3(filepath)
            wav_filepath = filepath + ".wav"
            wav.export(wav_filepath, format='wav', bitrate="96k")
            y, sr = librosa.load(wav_filepath, sr=sr)
            os.remove(wav_filepath)
        else:
            y, sr = librosa.load(filepath, sr=sr)
    except Exception as e:
        print(f"{filepath}: error while processing audio file: {e}")
        exit()
    
    # Get hyperparam for mp3 to spectrogram
    hop_length = kwargs['hop_length']
    win_length = kwargs['win_length']
    n_fft = kwargs['n_fft']
    n_mels = kwargs['n_mels']
    pwr_spect = kwargs['power_spectrogram']
    fmin = kwargs['fmin']
    fmax = kwargs['fmax']
    
    # TODO Uh taken from github... will change when I understand how to use librosa 
    
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                            win_length=win_length)) ** pwr_spect
    
    S = librosa.feature.melspectrogram(S=D, n_fft=n_fft, n_mels=n_mels,
                                       fmin=fmin, fmax=fmax)
    
    dbs = librosa.core.power_to_db(S)

    spectro = np.squeeze(dbs).T
    
    return spectro

"""
Returns the audio data for a spectrogram, consisting of a sequence of "snippets" from <spectro> around each snap. 
If a snap occurs at x ms, the snippet around that snap is the interval (x-WINDOW_SIZE, x+WINDOWSIZE).
"""
def get_audio_data(spectro, bar_len, offset):
    num_snaps = get_num_snaps(spectro, bar_len, offset)
    data = np.zeros((num_snaps, 2 * WINDOW_SIZE + 1, spectro.shape[1]))
    padded_spectro = np.pad(spectro, pad_width=((WINDOW_SIZE, WINDOW_SIZE+offset), (0,0)), constant_values=np.min(spectro))
    
    for snap_num in range(num_snaps):
        ms = snap_to_ms(bar_len, offset, snap_num)
        data[snap_num] = padded_spectro[ms+offset:ms+2*WINDOW_SIZE+1+offset]
        
    return data

"""
Given the filepath to a .osu file, returns:
- a list of notes in the song (see format below)
- bar_len
- offset

Each note (with snap timing <snap> and type <type>) in the .osu file is represented as (<snap>, <type>).
The list of notes will contain a sequence of these (<snap>, <type>) tuples.

Returns None, None, None and prints an error if the map satisfies one of the following:
- Are not in Taiko mode
- Exceeds maximum snap length
- Contains variable BPMs
- Contains an offset that is a float
- Contains an offset that is negative (before the audio file starts)
- Contains notes that don't fall on a snap
"""
def get_map_notes(filepath):
    with open(filepath, encoding="utf8") as file:
        osu = file.readlines()
    i = 0
    while (i < len(osu)):
        if osu[i][:4] == "Mode" and osu[i][-2] != "1": # Taiko mode maps have "Mode: 1"
            print(f"{os.path.basename(filepath)}: Wrong mode")
            return None, None, None
        if osu[i] == "[TimingPoints]\n":
            timing_points_index = i 
            while (i < len(osu) and osu[i] != "\n"):
                i += 1
            timing_points_endpoint = i
        if osu[i] == "[HitObjects]\n":
            hit_objects_index = i
            while (i < len(osu) and osu[i] != "\n"):
                i += 1
            hit_objects_endpoint = i
        i += 1
    timing_points = osu[timing_points_index+1: timing_points_endpoint] 
    hit_objects = osu[hit_objects_index+1: hit_objects_endpoint]
    
    # set offset and bar_len according to first timing point
    bar_len = float(timing_points[0].split(",")[1])
    try:
        offset = int(timing_points[0].split(",")[0])
    except:
        print(f"{os.path.basename(filepath)}: Found float offset")
        return None, None, None
    
    if offset < 0:
        print(f"{os.path.basename(filepath)}: Found negative offset")
        return None, None, None

    # check for extra uninherited timing points
    for timing_point in timing_points[1:]:
        # timing_point format:
        # time,beatLength,meter,sampleSet,sampleIndex,volume,uninherited,effects
        ary = timing_point.split(",")
        if ary[6] == "1": 
            # extra uninherited timing point, invalid .osu
            print(f"{os.path.basename(filepath)}: Found multiple uninherited timing points")
            return None, None, None

    # Returns the snap and the type of hit_object.
    # Returns None if hit_object has an invalid snap.
    def get_note_snap_type(hit_object):
        ary = hit_object.split(",")
        snap_num = ms_to_snap(bar_len, offset, int(ary[2]))
        if snap_num == None:
            # note is not snapped, invalid .osu
            print(f"{os.path.basename(filepath)}: Found unsnapped note")
            return None
        elif snap_num > MAX_SNAP:
            print(f"{os.path.basename(filepath)}: Map exceeds max snap limit")
            return None
        else:
            return snap_num, get_note_type(ary[4])


    # Check if max snaps exceeded by last hit object
    if (get_note_snap_type(hit_objects[-1]) is None):
        return None, None, None

    # Add all hit objects into lst, and return
    lst = []
    for hit_object in hit_objects: 
        snap_type = get_note_snap_type(hit_object)
        if (snap_type is None):
            return None, None, None
        lst.append((snap_type[0], snap_type[1]),) 
    return lst, bar_len, offset

"""
Note: DEPRACATED
Given a path to a mapset, a path_dict (see create_directory() in datasets.py), and a difficulty
name (from 'Kantan', 'Futsuu', 'Muzukashii', 'Oni'), attempt to extract and return the following:
- Audio data for the map, using get_audio_data()
- Map notes, using get_note_data()
- bar_len
- offset

Returns None, None, None, None upon failure.

Note: DEPRACATED
"""
def get_map_data(path, path_dict, diff):
    
    audio = path_dict["audio"]
    diff_path = os.path.join(path, path_dict[diff])
    
    notes, bar_len, offset = get_map_notes(diff_path)
    if notes is None:
        return None, None, None, None
        
    map_audio = get_map_audio(os.path.join(path, audio))
    num_snaps = get_num_snaps(map_audio, bar_len, offset)
    
    notes_data = get_note_data(notes, num_snaps)
    audio_data = get_audio_data(map_audio, bar_len, offset)
    
    return audio_data, notes_data, bar_len, offset

"""
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
        
"""
Produces a PyPlot of the spectrogram spectro.
"""
def plot_spectrogram(spectro):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(spectro.T)
    ax.set_aspect(0.5/ax.get_data_ratio(), adjustable='box')
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == "__main__":
    test = False
    # create testing variables
    if test:
        spectro = get_map_audio("test.mp3")
        notes, bar_len, offset = get_map_notes("test.osu")
        num_snaps = get_num_snaps(spectro, bar_len, offset)
        notes_data = get_note_data(notes, num_snaps)
        audio_data = get_audio_data(spectro, bar_len, offset)
        notes_src = torch.unsqueeze(torch.Tensor(notes_data), 0)
        notes_src = torch.tile(torch.unsqueeze(notes_src, 3), (1,1,1,4))
        audio_src = torch.unsqueeze(torch.Tensor(audio_data), 0)
        mask = torch.zeros(audio_src.shape[1], audio_src.shape[1])