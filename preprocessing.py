import os, pickle
import numpy as np
import json
from preprocessing_helpers import *
from helper import filter_to_snaps

data_directory = "data"
pickle_data_path = os.path.join(data_directory, "data.pkl") # file for all song names and difficulties
npy_data_directory = os.path.join(data_directory, "npy") # directory for all preprocessed numpy data


years = ["2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
diffs = ["kantan", "futsuu", "muzukashii", "oni"]

songs = {}

"""
Finds all mapsets in the data/20** folders that contain all of the following difficulty names:
"Kantan", "Futsuu", "Muzukashii", "Oni"
For each such mapset, store the audio and .osu filenames in a dictionary:
{'audio': <audio_file.mp3>, 'kantan': <kantan.osu>, ...}
Store all such dictionaries in data/data.pkl.
"""
def create_path_dict():
    
    total_sets = 0
    total_diffs = 0
    
    for year in years:
        num_sets = 0
        num_diffs = 0
        year_path = os.path.join(data_directory, year)
        if not (os.path.isdir(year_path)):
            continue
        files = os.listdir(year_path)
        for song in files:
            song_path = os.path.join(year_path, song)
            song_dir = os.listdir(song_path)
            
            path_dict = {"audio": None}
            for diff in diffs:
                path_dict[diff] = None
            
            # Find audio and difficulties
            for file in song_dir:
                if file.endswith(".mp3"):
                    path_dict["audio"] = file
                for diff in diffs:
                    if (f"[{diff}].osu" in file.lower() or
                        f"'s {diff}].osu" in file.lower()):
                        path_dict[diff] = file
            
            # Save filepaths
            if path_dict["audio"] is not None:
                for diff in diffs:
                    if path_dict[diff] is None:
                        path_dict.pop(diff)
                        
                if len(path_dict) > 1:
                    num_sets += 1
                    total_sets += 1
                    songs[song_path] = path_dict
                    
                    num_diffs += len(path_dict) - 1
                    total_diffs += len(path_dict) - 1
    
        print(f"[{year}] Mapsets: {num_sets}, Difficulties: {num_diffs}")
    
    with open(pickle_data_path, "wb") as out_file:
        pickle.dump(songs, out_file)
        
    print(f"Total Mapsets: {total_sets}, Total Difficulties:  {total_diffs}")


"""
Processes all mapsets in data/data.pkl (see create_path_dict()) into numpy data, and stores
the numpy data in data/npy/.

For each mapset in data/data.pkl, if the mapset has any valid difficulties:
- Create a folder data/npy/audio/data <mapset_folder_name>/, and store the audio's numpy
  as audio_data.npy in the newly-created folder.
- For each valid difficulty, create a folder data/npy/<difficulty_name>/data <mapset_folder_name>/, 
  and store the notes' data in notes_data.npy, and timing data (including the length of one bar in ms,
 and offset (time of first beat in ms)) in timing.npy.

force=True allows overwriting of existing folders/data in data/npy/ if needed.
"""
def create_data(force=False):
    
    total_diffs = {}
    total_ms = {}
    for diff in diffs:
        total_diffs[diff] = 0
        total_ms[diff] = 0
    
    with open(pickle_data_path, "rb") as in_file:
        songs = pickle.load(in_file)
        
    for path in songs:
        path_dict = songs[path]
        audio_filename = path_dict["audio"]
        # get map audio only if there is a valid difficulty, as this is time consuming
        map_audio = None
        audio_directory = os.path.join(npy_data_directory, "audio", path.replace("\\", " "))
        
        for diff in diffs:
            diff_directory = os.path.join(npy_data_directory, diff, path.replace("\\", " "))
            if (not os.path.exists(diff_directory)) or force:
                if diff in path_dict:
                    diff_path = os.path.join(path, path_dict[diff])
                    notes, offset, bar_len = get_map_data(diff_path)
                    timing_data = {
                        "offset": offset,
                        "bar_len": bar_len
                    }
                    if notes is not None:
                        if map_audio is None:
                            map_audio = get_map_audio(os.path.join(path, audio_filename))
                        if map_audio is not None:
                            notes_data = get_note_data(notes, map_audio.shape[0])
                            notes_data = filter_to_snaps(notes_data, bar_len, offset, 2, numpy=True)
                            print(f"{os.path.basename(path)} [{diff}]: Saving...")
                            total_diffs[diff] += 1
                            total_ms[diff] += map_audio.shape[0]
                            if not os.path.exists(diff_directory):
                                os.makedirs(diff_directory)
                            if not os.path.exists(audio_directory):
                                os.makedirs(audio_directory)
                            np.save(os.path.join(audio_directory, "audio_data.npy"), map_audio) 
                            np.save(os.path.join(diff_directory, "notes_data.npy"), notes_data)
                            with open(os.path.join(diff_directory, "timing_data.json"), "w+") as file:
                                json.dump(timing_data, file)
    
    for diff in diffs:
        print(f"Total Valid {diff} Difficulties: {total_diffs[diff]}")
        print(f"Total {diff} Time Steps: {total_ms[diff]}")


if __name__ == "__main__":
    create_path_dict()
    create_data()