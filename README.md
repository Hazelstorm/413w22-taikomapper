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

TaikoMapper is a modular seq2seq model that produces *osu!Taiko* maps. TaikoMapper takes in a preprocessed (see the paragraphs below) audio file, and outputs a time series of *Taiko* notes. 

To preprocess an audio file, we require the ```BPM``` and ```offset``` of the song (note that TaikoMapper only supports songs that don't have varying tempos). With inspiration from [*Osu! Beatmap Generator*](https://github.com/Syps/osu_beatmap_generator), the audio is first converted into a [mel spectrogram](https://en.wikipedia.org/wiki/Mel_scale) using [```librosa```](https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html). The mel scale divides the total range of frequencies (```fmin``` and ```fmax``` in ```hyper_param.py```) into frequency bands of equal logarithmic length. The spectrogram produced by this procedure has shape ```L x n_mels```, where ```L``` is the length of the audio file (in milliseconds) and ```n_mels``` (specified in ```hyper_param.py```) is the number of frequency bands (or "mel"s) in the frequency range. 

Then, we extract a time window from the spectrogram around each "snap", and place the time windows into a sequence.  A "snap" is a time point on the boundary of each 1/4 regular subdivision of a measure/bar, the length of one bar being 60000/BPM milliseconds (for example, in a 100BPM song, each bar is 600ms in length, so each snap is 150ms apart. If a bar begins at time 1000ms, the snaps occur at 1000ms, 1150ms, 1300ms, 1450ms, ...). The time window size is specified by ```window_size``` in ```hyper_params.py```; we extract ```window_size``` ms to the left and right of each snap from the spectrogram. The resulting sequence of time windows has size ```N x (2 * window_size + 1) x n_mels```, with ```N``` being the total number of snaps in the audio file. We flatten the spectrogram windows to instead have size ```N x ((2 * window_size + 1) * n_mels)```.

Taikomapper receives this ```N x ((2 * window_size + 1) * n_mels)``` sequence of spectrogram windows around each snap, and produces a sequence of length ```N```, with each entry being an integer from 0-4. Each entry determines the note (or the absence of a note) occurring on its respective snap, with 0, 1, 2, 3, and 4 respectively indicating no note, don, kat, don finisher, and kat finisher.


## Model

TaikoMapper's is constructed using three seq2seq models that feed into each other. The three seq2seq models are:
- ```notePresenceRNN```, used to determine where the taiko notes are placed (ignoring note type). ```notePresenceRNN``` accepts the ```N x ((2 * window_size + 1) * n_mels)``` sequence of spectrogram windows, and outputs a sequence of ```N``` floats, with positive value indicating that a note should be placed on the respective snap. ```notePresenceRNN``` processes the spectrogram windows through a linear embedding layer, a GRU, and then a linear layer with one output. The embeddings have size ```notePresenceRNN_embedding_size```, and the GRU's hidden unit has size ```notePresenceRNN_RNN_hidden_size```. On a single snap of input (one spectrogram window), the architecture looks like this:

<p align="center">
  <img src="/images/notePresenceRNN.png" alt="notePresenceRNN Architecture" width="600"/>
</p>

- ```noteColourRNN```, used to assign colour to the uncoloured sequence of notes produced by ```notePresenceRNN```. ```noteColour``` takes in the ```N x ((2 * window_size + 1) * n_mels)``` spectrogram windows, and ```notes_data```, a binary sequence of length ```N``` indicating where notes are present (0 for absent, 1 for present). ```noteColourRNN``` outputs a sequence of ```N``` floats, with positive value indicating that the note on the snap should be coloured blue (kat), and negative value indicating that a note should be coloured red. Note that ```notePresenceRNN``` is forced to predict 0 if there is no note occuring on the snap. Internally, noteColourRNN concatenates ```notes_data``` to the spectrogram windows, and then again passes it through a linear embedding layer, a GRU, and then a linear layer. The architecture looks like this:

<p align="center">
  <img src="/images/noteColourRNN.png" alt="noteColourRNN Architecture" width="600"/>
</p>

- ```noteFinisherRNN```, used to assign whether a note is a finisher to an uncoloured sequence of notes produced by ```notePresenceRNN```. Architectually, ```notefinisherRNN``` is identical to ```noteColourRNN```: the inputs are the spectrogram windows and ```notes_data```, and the output is a sequence of ```N``` floats, with positive value to indicate finisher.

The following diagram gives an overview on how the three models are connected.
<p align="center">
  <img src="/images/TaikoMapper.png" alt="TaikoMapper Architecture" width="600"/>
</p>

Each of the three seq2seq models can be trained separately; see ```train.py```.

## Preprocessing
