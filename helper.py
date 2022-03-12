
import math
import numpy as np

def snap_to_ms(bpm, offset, snap_num, snap_val=4):
    bar_len = 60000 / bpm
    ms = offset + (snap_num * bar_len / snap_val)
    return math.floor(ms)

def ms_to_snap(bpm, offset, ms, snap_val=4):
    bar_len = 60000 / bpm
    snap_num = (ms - offset) / (bar_len / snap_val)
    error = abs(snap_num - round(snap_num))
    if error < 0.05:
        return round(snap_num)

def get_map_data(filepath):
    """
    Given the filepath to a .osu, returns a numpy matrix
    representing the notes in the song. 
    """
    pass
    
def get_audio_data(filepath, window_size, bpm, offset, snap_val=4):
    """
    Given the filepath to a .mp3 and a window size, returns a numpy array
    with spectrogram data about the song.
    
    The mp3 is sampled at every snap with additional data on each side of
    the point with size `window_size`
    """
    pass

if __name__ == "__main__":
    for bpm in range(50, 200):
        for snap_num in range(100):
            ms = snap_to_ms(bpm, 0, snap_num)
            assert(ms_to_snap(bpm, 0, ms) == snap_num)
            assert(ms_to_snap(bpm, 0, ms+20) is None)
            assert(ms_to_snap(bpm, 0, ms-20) is None)


