import sys
import librosa
import math
import numpy as np
from hyper_param import *

SNAP = get_snap()

def snap_to_ms(bar_len, offset, snap_num, snap_val=SNAP):
    ms = offset + (snap_num * bar_len / snap_val)
    return math.floor(ms)

def ms_to_snap(bar_len, offset, ms, snap_val=SNAP):
    snap_num = (ms - offset) / (bar_len / snap_val)
    error = abs(snap_num - round(snap_num))
    if error < 0.05:
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
    return is_kat + 2 * is_finish

def get_num_snaps():
    pass

def make_map_data_array(data, num_snaps):
    pass

def get_map_data(filepath):
    """
    Given the filepath to a .osu, returns a numpy matrix
    representing the notes in the song. 
    """
    with open(filepath, encoding="utf8") as file:
        osu = file.readlines()
    i = 0
    count = 0
    while (i < len(osu)):
        # osu[timing_points+1:hit_objects]
        # osu[hit_objects+1:]
        if osu[i] == "[TimingPoints]\n":
            timing_points_index = i 
            while (osu[i] != "\n"):
                i += 1
            timing_points_endpoint = i
        if osu[i] == "[HitObjects]\n":
            hit_objects_index = i
        i += 1
    timing_points = osu[timing_points_index+1: timing_points_endpoint] # -1 should be good
    hit_objects = osu[hit_objects_index+1:-1]
    
    # set offset and bar_len according to first timing point
    offset = int(timing_points[0].split(",")[0])
    bar_len = float(timing_points[0].split(",")[1])

    # check for extra uninherited timing points
    for timing_point in timing_points[1:]:
        # time,beatLength,meter,sampleSet,sampleIndex,volume,uninherited,effects
        ary = timing_point.split(",")
        if ary[6] == "1": 
            # extra uninherited timing point, invalid .osu
            print("Found multiple uninherited timing points, exiting")
            return None
    
    # assume the bar_len and offset variables are already set pls
    # you need to check that every note falls on 1/4 snap
    # the ms_to_snap function should return None if the note is invalid
    lst = []
    for hit_object in hit_objects: 
        # x,y,time,type,hitSound,objectParams,hitSample
        ary = hit_object.split(",")
        snap_num = ms_to_snap(bar_len, offset, int(ary[2]))
        if snap_num == None:
            # note is not snapped, invalid .osu
            print("Found unsnapped note, exiting")
            return None
        else:
           lst.append((snap_num, get_note_type(ary[4])),) 
    return lst
    
def get_audio_data(filepath, kwargs=get_mel_param()):  
    """
    Given the filepath to a .mp3 and a window size, returns a numpy array
    with spectrogram data about the song.
    
    The mp3 is sampled at every snap with additional data on each side of
    the point with size `window_size`
    """
    sr = kwargs['sr']
    try: 
        y, sr = librosa.load(filepath, sr)
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

    spectrogram = np.squeeze(dbs).T
    
    return spectrogram


if __name__ == "__main__":
    for bpm in range(50, 200):
        bar_len = 60000 / bpm
        for snap_num in range(100):
            ms = snap_to_ms(bar_len, 0, snap_num)
            assert(ms_to_snap(bar_len, 0, ms) == snap_num)
            assert(ms_to_snap(bar_len, 0, ms+20) is None)
            assert(ms_to_snap(bar_len, 0, ms-20) is None)
      
# print(get_audio_data("test.mp3"))
get_map_data("test.osu")

# matrx = get_audio_data("test2.mp3")
# print(matrx.shape)