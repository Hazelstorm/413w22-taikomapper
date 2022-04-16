# TaikoMapper

*Dependencies*: Python 3.8, librosa, ffmpeg, pydub, PyTorch

## osu!Taiko
[*osu!*](https://osu.ppy.sh/) is a free-to-play rhythm game in which players attempt to click on circles to the beat of the music. In *osu!Taiko*, a game mode in *osu!* inspired by *Taiko no Tatsujin*, players are presented with a sequence of incoming red and blue circles (called "don"s and "kat"s respectively) approaching a drum. When a don or kat arrives at the center of the drum, the player must use the keyboard to tap the drum in the center (default keys X or C) or on the edge (Z or V) according to the colour of the note. Both don's and kat's also have a respective "finisher" variant (indicated by a larger circle of the respective colour), requiring the player to tap both center or both edge keys. 

A sample gameplay video (by contributor [Hazelstorm](https://github.com/Hazelstorm)) can be found [here](https://www.youtube.com/watch?v=7wP_YnOfpj8).

*Taiko* levels are stored in "beatmaps" (also known as "difficulties" or simply "maps"), and are community-created. Maps for the same song are grouped into "mapsets", which may be uploaded to the *osu!* website. Each *Taiko* mapset contains an audio file (typically MP3) and one ".osu" file for each difficulty. The .osu file format is in human-readable, and contains the following information:
- Filename of the audio file.
- Song/mapset/difficulty metadata, such as song title, song author, beatmap author, difficulty name, and gamemode. The ```Mode``` field specifies the gamemode; ```Mode: 1``` is used for *osu!Taiko* maps.
- Gameplay difficulty parameters, such as ```OverallDifficulty``` which determines the precision at which the player needs to hit the notes (increasing ```OverallDifficulty``` decreases the time window for the player to hit notes).
- Aesthetic information, such as the filename of the map's background image.
- Timing information, including the BPM (tempo) and offset (time of first beat in milliseconds, relative to start of audio file) of the song. There could be multiple BPMs and offsets for a song with varying tempo. 
- The notes in the map. For *Taiko*, each note has an offset (relative to the start of the audio file) and a "type" (used to determine whether the note is a don/kat and whether the note is a finisher). 

Typically, a Taiko mapset has one or more of the following difficulty names, in order of increasing gameplay difficulty: "Kantan" (Easy), "Futsuu" (Normal), "Muzukashii" (Hard), "Oni" (Demon). However, mapset creators can give custom names to difficulties, especially for difficulties that are harder than Oni.

For more detailed information on the .osu file format, refer to the [*osu!Wiki*](https://osu.ppy.sh/wiki/en/Client/File_formats/Osu_%28file_format%29).

## Introduction

TaikoMapper is a modular seq2seq model that produces *osu!Taiko* maps. TaikoMapper takes in a preprocessed (see the paragraphs below) audio file, and outputs a time series of *Taiko* notes. 

To preprocess an audio file, we require the ```BPM``` and ```offset``` of the song (note that TaikoMapper only supports songs that don't have varying tempos). With inspiration from [*Osu! Beatmap Generator*](https://github.com/Syps/osu_beatmap_generator), the audio is first converted into a [mel spectrogram](https://en.wikipedia.org/wiki/Mel_scale) using [```librosa```](https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html). The mel scale divides the total range of frequencies (```fmin``` and ```fmax``` in ```hyper_param.py```) into frequency bands of equal logarithmic length. The spectrogram produced by this procedure has shape ```L x n_mels```, where ```L``` is the length of the audio file (in milliseconds) and ```n_mels``` (specified in ```hyper_param.py```) is the number of frequency bands (or "mel"s) in the frequency range. 

Then, we extract a time window from the spectrogram around each "snap", and place the time windows into a sequence.  A "snap" is a time point on the boundary of each 1/4 regular subdivision of a measure/bar, the length of one bar being 60000/BPM milliseconds (for example, in a 100BPM song, each bar is 600ms in length, so each snap is 150ms apart. If a bar begins at time 1000ms, the snaps occur at 1000ms, 1150ms, 1300ms, 1450ms, ...). The time window size is specified by ```window_size``` in ```hyper_params.py```; we extract ```window_size``` ms to the left and right of each snap from the spectrogram. The resulting sequence of time windows has size ```N x (2 * window_size + 1) x n_mels```, with ```N``` being the total number of snaps in the audio file. We flatten the spectrogram windows to instead have size ```N x ((2 * window_size + 1) * n_mels)```.

Taikomapper receives this ```N x ((2 * window_size + 1) * n_mels)``` sequence of spectrogram windows around each snap, and produces a sequence of length ```N```, with each entry being an integer from 0-4. Each entry determines the note (or the absence of a note) occurring on its respective snap, with 0, 1, 2, 3, and 4 respectively indicating no note, don, kat, don finisher, and kat finisher.


## Model

TaikoMapper's is constructed using three seq2seq models that feed into each other. The three seq2seq models are:
- ```notePresenceRNN```, used to determine where the taiko notes are placed (ignoring note type). ```notePresenceRNN``` accepts the ```N x ((2 * window_size + 1) * n_mels)``` sequence of spectrogram windows, and outputs a sequence of ```N``` floats, with positive value indicating that a note should be placed on the respective snap. ```notePresenceRNN``` processes the spectrogram windows through a linear embedding layer, a GRU, and then a linear layer with one output. The embeddings have size ```notePresenceRNN_embedding_size``` (found in ```hyper_params.py```), and the GRU's hidden unit has size ```notePresenceRNN_hidden_size```. On a single snap of input (one spectrogram window), the architecture looks like this:

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

### Number of model parameters
- ```notePresenceRNN```
  + The linear embedding layer takes in a spectrogram window of size ```(2 * window_size + 1) * n_mels```, and embeds it into a vector of size ```notePresenceRNN_embedding_size```, so there are ```((2 * window_size + 1)* n_mels + 1) * notePresenceRNN_embedding_size``` trainable parameters.
  + The GRU takes in an embedding of size ```notePresenceRNN_embedding_size```, and outputs a hidden unit of size ```notePresenceRNN_hidden_size```. Each GRU has three weight matrices of size ```(input_size, hidden_size)```, another three weight matrices of size ```(hidden_size, hidden_size)```, and three bias vectors of size ```hidden_size```. To compute the update gate, for example, we need to apply one of the ```(input_size, hidden_size)``` matrices to the input, one of the ```(hidden_size, hidden_size)``` matrices to the previous hidden unit, and add one of the bias vectors. The reset gate and candidate hidden unit also both need one of each. Thus, in total, we have ```3 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)``` trainable parameters in the GRU, where ```input_size``` is ```notePresenceRNN_embedding_size``` and ```hidden_size``` is ```notePresenceRNN_hidden_size```.
  + The fully-connected linear layer at the end has input size ```notePresenceRNN_hidden_size```, and output size 1, so there are ```notePresenceRNN_hidden_size + 1``` trainable parameters.
- ```noteColourRNN``` and ```noteFinisherRNN```
  + The linear embedding layer takes in a spectrogram window of size ```(2 * window_size + 1) * n_mels + 1```, and embeds it into a vector of size ```noteColourRNN_embedding_size```, so there are ```((2 * window_size + 1)* n_mels + 2) * noteColourRNN_embedding_size``` trainable parameters.
  + The GRU takes in an embedding of size ```noteColourRNN_embedding_size```, and outputs a hidden unit of size ```noteColourRNN_hidden_size```. So the GRU has ```3 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)``` trainable parameters in the GRU, where ```input_size``` is ```noteColourRNN_embedding_size``` and ```hidden_size``` is ```noteColourRNN_hidden_size```..
  + The fully-connected linear layer at the end has input size ```noteColourRNN_hidden_size```, and output size 1, so there are ```noteColourRNN_hidden_size + 1``` trainable parameters.


<!---
number of trainable parameters by default
--->

<!---
successful and unsuccessful example
--->


## Data and Preprocessing

### Source
We have downloaded a dump of "ranked" *Taiko* mapsets from [this osu! forum post](https://osu.ppy.sh/community/forums/topics/330552?n=1). An uploaded *osu!* mapset can become ranked after passing a quality assurance process. We chose to only use ranked mapsets from 2013-2021 to assure data quality, as older mapsets tend to have poorer quality due to the lax quality assurance criteria at the time. 

### Data Summary
There are a total of 2795 ranked mapsets from 2013-2021, with 9113 *Taiko* difficulties. 
<!--- total length, average length, total # snaps, average # snaps, total # notes, average # notes --->

### Preprocessing
First, create a ```data/``` directory in this repository's folder. In ```data/```, create the directories ```2013/, 2014/, 2015/, 2016/, 2017/, 2018/, 2019/, 2020/, 2021```.

The *Taiko* mapset dump categorizes the mapsets by year. Each mapset is in .osz format, used for compressed *osu!* mapsets. To extract the mapsets using an already-existing installation of *osu!*, copy the .osz files into the ```osu!/Songs``` directory, and launch *osu!* (and go to the song selection screen). The extracted mapsets should be appear as folders in the ```osu!/Songs``` directory. Copy the mapset folders into ```data/20XX``` according to the mapset's year. We recommend this  process be done one year at a time.

Having all the mapsets in the ```data/``` directory, we run ```preprocessing.py```. ```preprocessing.py`` performs the following:
- ```create_path_dict()```: Create the file ```data.pkl```. For each mapset folder in ```data/```, find the audio file, and the .osu files. For any .osu file that corresponds to a Kantan, Futsuu, Muzukashii, or Oni difficulty, the .osu file's aboslute path and the audio file's absolute path is stored in ```data.pkl```.
- ```create_data()```: Reading from ```data.pkl```, each mapset from ```data.pkl``` has its audio file converted into a mel spectrogram (not yet converted to spectrogram windows), as described in the [Introduction](#introduction) section. Each difficulty in the mapset is converted into a time series of ```N``` notes (```N``` being the number of snaps in the song), and also has its timing data (BPM, stored as ```bar_len = 60000/BPM```) and offset extracted. Both the spectrogram and notes time series are numpy arrays; these numpy arrays are dumped into the ```data/npy``` directory. The BPM and offest are stored in a json file. 

The conversion from spectrogram to spectrogram windows is performed during training time, as we wanted the ability to change the hyperparameter ```window_size``` without preprocessing.

### Data Split
We allocated 80% of the mapsets for training, 10% of the mapsets for validation, and 10% of the mapsets for testing. This is because our evaluation will be mostly qualitative; there is no objective criteria to distinguish correct and incorrect generated maps, and our loss function does not fully capture the quality of our model. Also, the validation loss was only computed every 10 epochs of training, to reduce training time. 

## Quantitative Measures
We have defined different loss measures for the three models. 
- For ```notePresenceRNN```, recall that its output of ```N``` floats indicate the note presence at that snap. The greater the float value, the more confidence the model has in placing a note at that snap. On the other hand, the ground truth is a sequence of ```N``` 0s or 1s, indicating whether there is actually a note at that snap in the map. Thus, we've decided to take the softmax of the model output, to convert the output sequence into a sequence of probabilities on whether there is a note at the snap. Then, we compute the binary cross-entropy loss of this probability sequence with the ground truth. These two operations are combined into a [binary cross-entropy with logits](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) loss. However, in a typical *Taiko* map, most snaps do not have notes; there are usually around 3-12 times more empty snaps than snaps with notes. The hyperparameter ```note_presence_weight``` (in ```hyper_params.py```) is used to compensate for this note sparsity, through scaling the weight positive examples by ```note_presence_weight```.
- For ```noteColourRNN```, its output of ```N``` floats indicate whether the note at that snap should be coloured blue. The greater the value, the more likely the note is to be blue. The ground truth is a sequence of integers from 0 to 4; recall that 0, 1, 2, 3, and 4 represent no note, don, kat, don finisher, and kat finisher respectively. If the ground truth is 2 or 4 (indicating a blue/kat note), then ```noteColourRNN``` should be penalized for predicting a red note (low value); if the ground truth is 1 or 3 (indicating a red/don note), the model should be penalized for predicting a blue note (high value). If the ground truth is 0, the model is forced to predict 0, as mentioned before. Again, we use the binary cross-entropy with logits loss here; however we filter out the sequence entries that represent no note, as we don't want to penalize the model's colour prediction when a note is not present. Using binary CE with logits, we compare this filtered output with a binary ground-truth sequence of the same length, with 0 for red and 1 for blue. This time, weighing a positive example is not necessary; the number of red and blue notes in a typical Taiko map are similar.
- For ```noteFinisherRNN```, its output of ```N``` floats indicate whether the note at that snap should be a finisher note. The greater the value, the more likely the note is to be a finisher. Again, the ground truth is a sequence of integers from 0 to 4; 3 and 4 indicate a finisher note, while 1 and 2 indicate a non-finisher. We perform the same filtering operation as in ```noteColourRNN```'s loss, and use binary CE with logits. However, since finisher notes are relatively rare, we scale the weight of the positive examples by ```note_finisher_weight```.


## Hyperparameters
Our TaikoMapper model has the following hyperparameters:
- ```notePresenceRNN_embedding_size```, ```notePresenceRNN_hidden_size```, ```noteColourRNN_embedding_size```, and ```noteColourRNN_hidden_size```, as explained in the [Model](#model) section.  
- ```n_mels```, ```window_size```, ```fmin```, ```fmax```: These hyperparameters determine the information stored in the spectrogram windows.

In addition, the training loop has the following hyperparameters:
- ```note_presence_weight``` (in ```hyper_params.py```): In a typical *Taiko* map, most snaps do not have notes. ```note_presence_weight``` is used to compensate for this note sparsity by emphasizing on present notes when computing loss for ```notePresenceRNN```.
- ```learning_rate```, ```wd``` (weight decay): These hyperparameters can be found in ```train.py```.

### Tuning
To tune the learning rate, we've tried training the ```notePresenceRNN``` for 100 epochs with varying learning rates (```1e-4, 1e-5, 1e-6```). After plotting the training curves, we've decided to use a learning rate of ```1e-5``` for ```notePresenceRNN```.

<p align="center">
  <img src="/images/learning_rate_training_curves.png" alt="Learning Rate Training Curves" width="600"/>
</p>
