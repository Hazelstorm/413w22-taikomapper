import librosa, math, os
import numpy as np
import matplotlib.pyplot as plt
import hyper_param
from pydub import AudioSegment # to convert mp3 to wav

"""
Return whether bit i is set in x.
"""
def bit_flag(x, i):
    return bool(x & (1 << i))

"""
Converts an integer <note>, representing the note type in .osu files (see below), to
an integer representing the note type in our model.

In .osu files, note types are stored in a 4-bit integer. If either the 1st or 3rd bit is set,
the note is a kat; otherwise the note is a don. If the 2nd bit is set, the note is a finisher.
The 0th bit does not matter. This gives the following representations of Taiko notes:
don: 0000 or 0001 (0 or 1)
kat: 0010 or 0011 or 1000 or 1001 or 1010 or 1011 (2 or 3 or 8 or 9 or 10 or 11)
don finisher: 0100 or 0101 (4 or 5)
kat finisher: 0110 or 0111 or 1100 or 1101 or 1110 or 1111 (6 or 7 or 12 or 13 or 14 or 15)

In our model, note types are stored as an integer between 1 and 4, as follows:
none: 0
don: 1
kat: 2
don finisher: 3
kat finisher: 4
"""
def get_note_type(note):
    is_kat = bit_flag(int(note), 1) or bit_flag(int(note), 3)
    is_finish = bit_flag(int(note), 2)
    return 1 + is_kat + 2 * is_finish

"""
Given the filepath to a .osu file, returns a list of notes in the song (see format below).

Each note (type <type> at time <time> in ms) in the .osu file is represented as (<time>, <type>).
The list of notes will contain a sequence of these (<time>, <type>) tuples.

Returns None and prints an error if the map isn't in Taiko mode.
"""
def get_map_data(filepath):
    with open(filepath, encoding="utf8") as file:
        osu_file = file.readlines()
    i = 0
    while (i < len(osu_file)):
        if osu_file[i][:4] == "Mode" and osu_file[i][-2] != "1": # Taiko mode maps have "Mode: 1"
            print(f"{os.path.basename(filepath)}: Wrong mode")
            return None, None, None
        if osu_file[i] == "[TimingPoints]\n":
            offset = float(osu_file[i+1].split(",")[0])
            bar_len = float(osu_file[i+1].split(",")[1])
            if offset < 0:
                print(f"{os.path.basename(filepath)}: Found negative offset")
                return None, None, None
        if osu_file[i] == "[HitObjects]\n":
            hit_objects_index = i
            while (i < len(osu_file) and osu_file[i] != "\n"):
                i += 1
            hit_objects_endpoint = i
        i += 1
    hit_objects = osu_file[hit_objects_index+1: hit_objects_endpoint]
    
    # Add all hit objects into lst, and return
    lst = []
    for hit_object in hit_objects: 
        ary = hit_object.split(",")
        time = int(ary[2])
        type = get_note_type(ary[4])
        lst.append((time, type))
    return lst, offset, bar_len

"""
Given a list of <notes> from get_map_notes(), create an array of length <song_len>, where 
the t'th entry indicates the note type found at time t ms in the song (including no note).

Note types are stored as an integer between 1 and 4, as follows:
none: 0
don: 1
kat: 2
don finisher: 3
kat finisher: 4
"""
def get_note_data(notes, song_len):
    data = np.zeros((song_len), dtype=np.ushort)
    for note in notes:
        if (0 <= note[0] and note[0] <= song_len-1):
           data[note[0]] = note[1]
    return data

"""
Convert a list of notes from get_note_data into one-hot vectors.
"""
def make_onehot(notes, num_classes=5): 
    I = np.eye(num_classes)
    return I[notes]

"""
Given the <filepath> to a .mp3 and a window size, returns a numpy array
containing spectrogram data for the .mp3 file.
The mp3 is sampled in a window of size 2 * WINDOW_SIZE around each snap.
convert=True converts mp3 files into wav before processing, as librosa doesn't
handle mp3 efficiently.
"""
def get_map_audio(filepath, convert=True): 
    sr = hyper_param.sr
    try: 
        if convert and filepath.endswith(".mp3"):
            wav = AudioSegment.from_mp3(filepath)
            wav_filepath = filepath + ".wav"
            wav.export(wav_filepath, format='wav', bitrate="96k")
            y, sr = librosa.load(wav_filepath, sr=sr)
            os.remove(wav_filepath)
        else:
            y, sr = librosa.load(filepath, sr=sr)
    except Exception as e:
        print(f"{filepath}: error while processing audio file: {e}")
        exit()

    # Get hyperparam for mp3 to spectrogram
    hop_length = hyper_param.hop_length
    win_length = hyper_param.win_length
    n_fft = hyper_param.n_fft
    n_mels = hyper_param.n_mels
    pwr_spect = hyper_param.power_spectrogram
    fmin = hyper_param.fmin
    fmax = hyper_param.fmax

    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                            win_length=win_length)) ** pwr_spect

    S = librosa.feature.melspectrogram(S=D, n_fft=n_fft, n_mels=n_mels,
                                       fmin=fmin, fmax=fmax)

    spectro = np.squeeze(S).T

    return spectro
        
"""
Produces a PyPlot of the spectrogram spectro.
"""
def plot_spectrogram(spectro):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(spectro.T)
    ax.set_aspect(0.5/ax.get_data_ratio(), adjustable='box')
    plt.gca().invert_yaxis()
    plt.show()