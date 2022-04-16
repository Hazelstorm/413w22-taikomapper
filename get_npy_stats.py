import os
from helper import get_npy_data
import numpy as np


dir_path = "Z:\\Users\\David\\Documents\\@@@@UTM\\2022 winter\\CSC 413\\p\\413w22-taikomapper\\data\\npy"
difficulties = ["futsuu","kantan","muzukashii","oni"]

def count_snaps_notes(path):
    audio_data, timing_data, notes_data = get_npy_data(path)
    t1, audioname = os.path.split(path)
    path_to_folder, difficulty_str = os.path.split(t1)
    total_snaps = np.shape(notes_data)[0]
    total_notes = np.count_nonzero(notes_data)
    length_of_song = np.shape(audio_data)[0]
    return (length_of_song, total_snaps, total_notes)

def get_count_matrix(dir_path):
    '''
    columns are the difficulties in order
    row 0: number of songs count
    row 1: length of all songs
    row 2: number of snaps
    row 3: number of notes
    '''
    num_difficulties = len(difficulties)
    counts_matrix = np.zeros((4, num_difficulties), dtype=int)
    for i in range(num_difficulties):
        print(difficulties[i])
        diff_dir_path = os.path.join(dir_path, difficulties[i])
        os.chdir(diff_dir_path)
        subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
        for song in subdirs:
            song_dir_path = os.path.join(diff_dir_path, song)
            # print(song_dir_path)
            song_len, num_snaps, num_notes = count_snaps_notes(song_dir_path)
            counts_matrix[0][i] += 1
            counts_matrix[1][i] += song_len
            counts_matrix[2][i] += num_snaps
            counts_matrix[3][i] += num_notes
            if (counts_matrix[0][i] % 100 == 0):
                print(counts_matrix[0][i])
    return counts_matrix

cm = get_count_matrix(dir_path)
print(cm)

