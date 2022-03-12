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
    if error < 0.015:
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

def get_map_data(filepath):
    """
    Given the filepath to a .osu, returns a numpy matrix
    representing the notes in the song. 
    """
    with open(filepath) as file:
        osu = file.readlines()
    i = 0
    count = 0
    while (i < len(osu)):
        # osu[timing_points+1:hit_objects]
        # osu[hit_objects+1:]
        if osu[i] == "[TimingPoints]\n":
            timing_points_index = i #? this should work
        if osu[i] == "[HitObjects]\n":
            hit_objects_index = i

    timing_points = osu[timing_points_index+1:hit_objects_index] 
    hit_objects = osu[hit_objects_index+1:]
    
    for timing_point in timing_points:
        # time,beatLength,meter,sampleSet,sampleIndex,volume,uninherited,effects
        ary = timing_point.split(",")
        
    for hit_object in hit_objects: 
        # x,y,time,type,hitSound,objectParams,hitSample
        ary = hit_object.split(",")
        
def get_audio_data(filepath, kwargs=get_mel_param()):
    """
    Given the filepath to a .mp3 and a window size, returns a numpy array
    with spectrogram data about the song.
    
    The mp3 is sampled at every snap with additional data on each side of
    the point with size `window_size`
    """
    
    try: 
        y, sr = librosa.load(filepath)
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

    spectrogram = np.squeeze(dbs)
    
    return spectrogram.T


if __name__ == "__main__":
    for bpm in range(50, 200):
        bar_len = 60000 / bpm
        for snap_num in range(100):
            ms = snap_to_ms(bar_len, 0, snap_num)
            assert(ms_to_snap(bar_len, 0, ms) == snap_num)
            assert(ms_to_snap(bar_len, 0, ms+10) is None)
            assert(ms_to_snap(bar_len, 0, ms-10) is None)


# print(get_audio_data("test.mp3"))
get_map_data("./data/2021/203283 Mitchie M - Birthday Song for Miku/Mitchie M - Birthday Song for Miku (Krisom) [Dekaane's Oni].osu")