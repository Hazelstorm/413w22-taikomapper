
import os, pickle
import numpy as np
from tqdm import tqdm
from helper import *

years = ["2008", "2009", "2010", "2011", "2012", "2013", "2014", 
         "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
songs = {}

"""
Finds all mapsets in the data/20** folders that contain all of the following difficulty names:
"Kantan", "Futsuu", "Muzukashii", "Oni"
For each such mapset, store the audio and .osu filenames in a dictionary:
{'audio': <audio_file.mp3>, 'kantan': <kantan.osu>, ...}
Store all such dictionaries in data/data.pkl.
"""
def create_directory():
    
    total_sets = 0
    
    for year in years:
        num_sets = 0
        year_path = os.path.join("data", year)
        if not (os.path.isdir(year_path)):
            continue
        files = os.listdir(year_path)
        for song in files:
            song_path = os.path.join(year_path, song)
            song_dir = os.listdir(song_path)
            
            audio, kantan, futsuu, muzu, oni = [None]*5
            
            # Find audio and difficulties
            for file in song_dir:
                if ".mp3" in file:
                    audio = file
                if ("[kantan].osu" in file.lower() or
                    "'s kantan].osu" in file.lower()):
                    kantan = file
                if ("[futsuu].osu" in file.lower() or
                    "'s futsuu].osu" in file.lower()):
                    futsuu = file
                if ("[muzukashii].osu" in file.lower() or
                    "'s muzukashii].osu" in file.lower()):
                    muzu = file
                if ("[oni].osu" in file.lower() or
                    "'s oni].osu" in file.lower()):
                    oni = file
            
            # Save filepaths
            if audio and kantan and futsuu and muzu and oni:
                num_sets += 1
                total_sets += 1
                songs[song_path] = {}
                songs[song_path]["audio"] = audio
                songs[song_path]["kantan"] = kantan
                songs[song_path]["futsuu"] = futsuu
                songs[song_path]["muzu"] = muzu
                songs[song_path]["oni"] = oni
    
        print(f"[{year}] Mapsets: {num_sets}")
    
    with open(os.path.join("data", "data.pkl"), "wb") as out_file:
        pickle.dump(songs, out_file)
        
    print(f"Total Mapsets:  {total_sets}")


"""
Processes all mapsets in data/data.pkl (see create_directory) into numpy data.
"""
def create_data(force=False):
    
    sets = 0
    time_steps = 0
    
    with open(os.path.join("data", "data.pkl"), "rb") as in_file:
        songs = pickle.load(in_file)
        
    for path in songs:
        directory = (os.path.join("data", "npy", os.path.basename(path)))
        if (not os.path.exists(directory)) or force:
            path_dict = songs[path]
            
            try:
                audio_data, notes_data, bar_len, offset = get_map_data(path, path_dict)
            except:
                print("Unknown exception")
                audio_data = None
            
            if audio_data is not None:
                print(f"Saving {path}")
                sets += 1
                time_steps += audio_data.shape[0]
                if not os.path.exists(directory):
                    os.makedirs(directory)
                np.save(os.path.join(directory, "audio_data.npy"), audio_data)
                np.save(os.path.join(directory, "notes_data.npy"), notes_data)
                np.save(os.path.join(directory, "timing.npy"), np.array([bar_len, offset]))
    
    print(f"Total Valid Mapsets: {sets}")
    print(f"Total Time Steps: {time_steps}")