# TaikoMapper

*Dependencies*: Python 3.8, librosa, ffmpeg, pydub, PyTorch

## osu!Taiko
[*osu!*](https://osu.ppy.sh/) is a free-to-play rhythm game in which players attempt to click on circles to the beat of the music. In *osu!Taiko*, a game mode in *osu!* inspired by *Taiko no Tatsujin*, players are presented with a sequence of incoming red and blue circles (called "don"s and "kat"s respectively) approaching a drum. When a don or kat arrives at the center of the drum, the player must use the keyboard to tap the drum in the center (default keys X or C) or on the edge (Z or V) according to the colour of the note. Both don's and kat's also have a respective "finisher" variant (indicated by a larger circle of the respective colour), requiring the player to tap both center or both edge keys. 

A sample gameplay video (by contributor [Hazelstorm](https://github.com/Hazelstorm)) can be found [here](https://www.youtube.com/watch?v=7wP_YnOfpj8).

*Taiko* levels are stored in "beatmaps" (also known as "difficulties" or simply "maps"), and are community-created. Maps for the same song are grouped into "mapsets", which may be uploaded to the *osu!* website. Each *Taiko* mapset contains an audio file (typically MP3) and one ".osu" file for each difficulty. The .osu file format is in human-readable, and contains the following information:
- Filename of the audio file.
- Song/mapset/difficulty metadata, such as song title, song author, beatmap author, and gamemode. The ```Mode``` field specifies the gamemode; ```Mode: 1``` is used for *osu!Taiko* maps.
- Gameplay difficulty parameters, such as ```OverallDifficulty``` which determines the precision at which the player needs to hit the notes (increasing ```OverallDifficulty``` decreases the time window for the player to hit notes).
- Aesthetic information, such as the filename of the map's background image.
- Timing information, including the BPM (tempo) and offset (time of first beat in milliseconds, relative to start of audio file) of the song. There could be multiple BPMs and offsets for a song with varying tempo. 
- The notes in the map. For *Taiko*, each note has an offset (relative to the start of the audio file) and a "type" (used to determine whether the note is a don/kat and whether the note is a finisher). 

For more detailed information on the .osu file format, refer to the [*osu!Wiki*](https://osu.ppy.sh/wiki/en/Client/File_formats/Osu_%28file_format%29).

## Introduction

*This section uses terminology specific to osu!Taiko. See the [osu!Taiko](#osutaiko) section for an introduction of osu!Taiko.*

TaikoMapper is a modular seq2seq model that produces *osu!Taiko* maps. TaikoMapper takes in a [preprocessed](preprocessing) audio file (.mp3, .wav, or .ogg), and outputs a time series of notes (from none, don, kat, don finisher, kat finisher), representing the note occurring on each snap.

TaikoMapper's modularity consists of three seq2seq models connected serially. The three seq2seq models are:
- notePresenceRNN. This

## Preprocessing
