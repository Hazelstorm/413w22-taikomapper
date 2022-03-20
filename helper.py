import sys, librosa, math, os
import numpy as np
import matplotlib.pyplot as plt
from hyper_param import *

SNAP = get_snap()
WINDOW_SIZE = get_window_size()
MAX_SNAP = get_max_snap()

def snap_to_ms(bar_len, offset, snap_num, snap_val=SNAP):
    ms = offset + (snap_num * bar_len / snap_val)
    return math.floor(ms)

def ms_to_snap(bar_len, offset, ms, snap_val=SNAP):
    snap_num = (ms - offset) / (bar_len / snap_val)
    error = abs(snap_num - round(snap_num))
    if error < 0.02 * SNAP:
        return round(snap_num)

def get_line(osu, header):
    for i, line in enumerate(osu):
        if header in line:
            return i

#Shamelessly stolen from A1 ¯\_(ツ)_/¯
def make_onehot(indicies, total=5): 
    """
    Convert indicies into one-hot vectors by
        1. Creating an identity matrix of shape [total, total]
        2. Indexing the appropriate columns of that identity matrix
    """
    I = np.eye(total)
    return I[indicies]

def bit_flag(x, i):
    return bool(x & (1 << i))

def get_note_type(note):
    is_kat = bit_flag(int(note), 1) or bit_flag(int(note), 3)
    is_finish = bit_flag(int(note), 2)
    return 1 + is_kat + 2 * is_finish

def get_num_snaps(spectro, bar_len, offset):
    return round((spectro.shape[0] - offset) // bar_len * SNAP)

def get_note_data(notes, num_snaps):
    data = np.zeros((num_snaps, 5))
    for note in notes:
        data[note[0], note[1]] = 1
        
    # If no note on tick, set Empty flag to true
    data[:,0] = data[:,0] + 1 - np.sum(data, axis=1)
    
    return data

def get_audio_data(spectro, bar_len, offset):
    num_snaps = get_num_snaps(spectro, bar_len, offset)
    data = np.zeros((num_snaps, 2 * WINDOW_SIZE + 1, spectro.shape[1]))
    padded_spectro = np.pad(spectro, ((WINDOW_SIZE, WINDOW_SIZE+offset), (0,0)))
    
    for snap_num in range(num_snaps):
        ms = snap_to_ms(bar_len, offset, snap_num)
        data[snap_num] = padded_spectro[ms+offset:ms+2*WINDOW_SIZE+1+offset]
        
    return data

def get_map_notes(filepath):
    """
    Given the filepath to a .osu, returns a numpy matrix
    representing the notes in the song. 
    """
    with open(filepath, encoding="utf8") as file:
        osu = file.readlines()
    i = 0
    while (i < len(osu)):
        if osu[i][:4] == "Mode" and osu[i][-2] != "1":
            print(f"Wrong mode in {filepath}, exiting")
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
        print(f"Found float offset in {filepath}, exiting")
        return None, None, None

    # check for extra uninherited timing points
    for timing_point in timing_points[1:]:
        # time,beatLength,meter,sampleSet,sampleIndex,volume,uninherited,effects
        ary = timing_point.split(",")
        if ary[6] == "1": 
            # extra uninherited timing point, invalid .osu
            print(f"Found multiple uninherited timing points in {filepath}, exiting")
            return None, None, None
    
    lst = []
    for hit_object in hit_objects: 
        # x,y,time,type,hitSound,objectParams,hitSample
        ary = hit_object.split(",")
        snap_num = ms_to_snap(bar_len, offset, int(ary[2]))
        if snap_num == None:
            # note is not snapped, invalid .osu
            print(f"Found unsnapped note at {ary[2]}ms, exiting")
            return None, None, None
        else:
           lst.append((snap_num, get_note_type(ary[4])),) 
    return lst, bar_len, offset
    
def get_map_audio(filepath, kwargs=get_mel_param()):  
    """
    Given the filepath to a .mp3 and a window size, returns a numpy array
    with spectrogram data about the song.
    
    The mp3 is sampled at every snap with additional data on each side of
    the point with size `window_size`
    """
    sr = kwargs['sr']
    try: 
        y, sr = librosa.load(filepath, sr=sr)
    except Exception:
        print("file error")
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

def get_map_data(path, path_dict):
    
    audio = path_dict["audio"]
    kantan = path_dict["kantan"]
    futsuu = path_dict["futsuu"]
    muzu = path_dict["muzu"]
    oni = path_dict["oni"]
    
    kantan_notes, bar_len, offset = get_map_notes(os.path.join(path, kantan))
    if kantan_notes is None:
        return None, None, None, None
    futsuu_notes, bar_len, offset = get_map_notes(os.path.join(path, futsuu))
    if futsuu_notes is None:
        return None, None, None, None
    muzu_notes, bar_len, offset = get_map_notes(os.path.join(path, muzu))
    if muzu_notes is None:
        return None, None, None, None
    oni_notes, bar_len, offset = get_map_notes(os.path.join(path, oni))
    if oni_notes is None:
        return None, None, None, None
        
    map_audio = get_map_audio(os.path.join(path, audio))
    num_snaps = get_num_snaps(map_audio, bar_len, offset)
    if num_snaps > MAX_SNAP:
        print("Audio exceeds max snap limit")
        return None, None, None, None
    
    kantan_data = get_note_data(kantan_notes, num_snaps)
    futsuu_data = get_note_data(futsuu_notes, num_snaps)
    muzu_data = get_note_data(muzu_notes, num_snaps)
    oni_data = get_note_data(oni_notes, num_snaps)
        
    audio_data = get_audio_data(map_audio, bar_len, offset)
    notes_data = np.stack([kantan_data, futsuu_data, muzu_data, oni_data], axis=0)
    
    return audio_data, notes_data, bar_len, offset
        

def plot_spectrogram(spectro):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(spectro.T)
    ax.set_aspect(0.5/ax.get_data_ratio(), adjustable='box')
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == "__main__":
    spectro = get_map_audio("test.mp3")
    notes, bar_len, offset = get_map_notes("test.osu")
    num_snaps = get_num_snaps(spectro, bar_len, offset)
    note_data = get_note_data(notes, num_snaps)
    audio_data = get_audio_data(spectro, bar_len, offset)
