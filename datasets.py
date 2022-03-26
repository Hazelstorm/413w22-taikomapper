
import os, pickle
import numpy as np
from tqdm import tqdm
from helper import *
import traceback

years = ["2008", "2009", "2010", "2011", "2012", "2013", "2014", 
         "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
diffs = ["kantan", "futsuu", "muzu", "oni"]
songs = {}

def create_directory():
    
    total_sets = 0
    total_diffs = 0
    
    for year in years:
        num_sets = 0
        num_diffs = 0
        year_path = os.path.join("data", year)
        files = os.listdir(year_path)
        for song in files:
            song_path = os.path.join(year_path, song)
            song_dir = os.listdir(song_path)
            
            path_dict = {"audio": None}
            for diff in diffs:
                path_dict[diff] = None
            
            # Find audio and difficulties
            for file in song_dir:
                if ".mp3" in file:
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
    
    with open(os.path.join("data", "data.pkl"), "wb") as out_file:
        pickle.dump(songs, out_file)
        
    print(f"Total Mapsets: {total_sets}, Total Difficulties:  {total_diffs}")

def create_data(force=False):
    
    total_diffs = {}
    time_steps = {}
    for diff in diffs:
        total_diffs[diff] = 0
        time_steps[diff] = 0
    
    with open(os.path.join("data", "data.pkl"), "rb") as in_file:
        songs = pickle.load(in_file)
        
    for path in songs:
        for diff in diffs:
            
            directory = os.path.join("data", "npy", diff, path.replace("\\", " "))
            audio_directory = os.path.join("data", "npy", "audio", path.replace("\\", " "))
            if (not os.path.exists(directory)) or force:
                path_dict = songs[path]
            
                if diff in path_dict:
                    
                    try:
                        audio_data, notes_data, bar_len, offset = get_map_data(path, path_dict, diff)
                    except Exception:
                        print(traceback.print_exc())
                        audio_data = None
                    
                    if audio_data is not None:
                        print(f"Saving {path} {diff}")
                        total_diffs[diff] += 1
                        time_steps[diff] += audio_data.shape[0]
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        if not os.path.exists(audio_directory):
                            os.makedirs(audio_directory)
                        np.save(os.path.join(audio_directory, "audio_data.npy"), audio_data)
                        np.save(os.path.join(directory, "notes_data.npy"), notes_data)
                        np.save(os.path.join(directory, "timing.npy"), np.array([bar_len, offset]))
    
    for diff in diffs:
        print(f"Total Valid {diff} Difficulties: {total_diffs[diff]}")
        print(f"Total {diff} Time Steps: {time_steps[diff]}")
        
