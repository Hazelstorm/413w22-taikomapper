
import os, pickle
import numpy as np
from helper import *

years = ["2021"]
songs = {}

def create_directory():
    num_kantan = 0
    num_futsuu = 0
    num_muzu = 0
    num_oni = 0
    
    for year in years:
        year_path = os.path.join("data", year)
        files = os.listdir(year_path)
        for song in files:
            song_path = os.path.join(year_path, song)
            song_dir = os.listdir(song_path)
            
            audio, kantan, futsuu, muzu, oni = [None]*5
            
            # Find audio and difficulties
            for file in song_dir:
                if ".mp3" in file:
                    audio = file
                if "Kantan].osu" in file or "'s Kantan].osu" in file:
                    kantan = file
                if "Futsuu].osu" in file or "'s Futsuu].osu" in file:
                    futsuu = file
                if "Muzukashii.osu" in file or "'s Muzukashii].osu" in file:
                    muzu = file
                if "Oni].osu" in file or "'s Oni].osu" in file:
                    oni = file
            
            if audio and (kantan or futsuu or muzu or oni):
                songs[song_path] = {}
                songs[song_path]["audio"] = audio
                if kantan:
                    num_kantan += 1
                    songs[song_path]["kantan"] = kantan
                if futsuu:
                    num_futsuu += 1
                    songs[song_path]["futsuu"] = futsuu
                if muzu:
                    num_muzu += 1
                    songs[song_path]["muzu"] = muzu
                if oni:
                    num_oni += 1
                    songs[song_path]["oni"] = oni
    
    print(f"Kantans: {num_kantan}")
    print(f"Futsuus: {num_futsuu}")
    print(f"Muzus: {num_muzu}")
    print(f"Onis: {num_oni}")
    
    with open("data.pkl", "wb") as out_file:
        pickle.dump(songs, out_file)
        
    return songs
        
def create_data():
    with open("data.pkl", "rb") as in_file:
        songs = pickle.load(in_file)
        
    for song in songs:
        pass
