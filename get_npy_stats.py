import os
from helper import get_npy_data
import numpy as np


dir_path = os.path.join("data", "npy")
difficulties = ["kantan", "futsuu", "muzukashii", "oni"]


"""
Returns the length of the song (in ms), total number of snaps in song, and 
total number of nonzero notes, in that order.
<path> is a filepath to a numpy difficulty file.
"""
def count_snaps_notes(path):
    audio_data, timing_data, notes_data = get_npy_data(path)
    total_snaps = np.shape(notes_data)[0]
    total_notes = np.count_nonzero(notes_data)
    length_of_song = np.shape(audio_data)[0]
    return (length_of_song, total_snaps, total_notes)

"""
Returns a dictionary containing informaiton on the following for each difficulty (and their totals):
- Total number of difficulties
- Total song length, in ms
- Total number of snaps
- Total number of notes
"""
def get_counts(dir_path):
    counts = {
        "Total": {"Number of Difficulties": 0,
            "Total Song length (ms)": 0,
            "Total snaps": 0,
            "Total notes": 0,}
    }
    for diff in difficulties:
        counts[diff] = {
            "Number of Difficulties": 0,
            "Total Song length (ms)": 0,
            "Total snaps": 0,
            "Total notes": 0,
        }
        diff_dir_path = os.path.join(dir_path, diff)
        subdirs = [d for d in os.listdir(diff_dir_path) if os.path.isdir(os.path.join(diff_dir_path, d))]
        for song in subdirs:
            song_dir_path = os.path.join(diff_dir_path, song)
            song_len, num_snaps, num_notes = count_snaps_notes(song_dir_path)
            counts[diff]["Number of Difficulties"] += 1
            counts[diff]["Total Song length (ms)"] += song_len
            counts[diff]["Total snaps"] += num_snaps
            counts[diff]["Total notes"] += num_notes
            counts["Total"]["Number of Difficulties"] += 1
            counts["Total"]["Total Song length (ms)"] += song_len
            counts["Total"]["Total snaps"] += num_snaps
            counts["Total"]["Total notes"] += num_notes

    return counts

counts = get_counts(dir_path)
for diff in counts.keys():
    print(f"{diff} statistics:")
    for attr in counts[diff].keys():
        print(f"{attr}: {counts[diff][attr]}")

