import os
from postprocessing_helpers import *
from preprocessing_helpers import get_map_audio
from helper import snap_to_ms, get_audio_around_snaps
import hyper_param

index_to_hitsound = { \
  # index in model, hitsound in .osu file
  1: 0, # don
  2: 2, # kat
  3: 4, # don finisher
  4: 6, # kat finisher
}

"""
Given a presence, colour, and finisher model, an audio file <audio_filepath>, and its bpm and offset, create a .osu file
by using the models to generate a Taiko map.
Parameters:
osu_filename: File name of the generated .osu file. If none, automatically choose a filename.
fields: an optional dictionary containing entries {field_name: value}, to set the field names in postprocessing_helpers.py
        such as "Title".
"""
def create_osu_file(presence_model, colour_model, finisher_model, audio_filepath, bpm, offset, osu_filename=None, fields={}):
    # See https://osu.ppy.sh/wiki/en/Client/File_formats/Osu_%28file_format%29
    bar_len = 60000/bpm

    print("Loading Audio...")
    spectro = get_map_audio(audio_filepath)
    print("Audio Loaded.")
    spectro = torch.tensor(spectro)
    audio_windows = get_audio_around_snaps(spectro, bar_len, offset, hyper_param.window_size)
    audio_windows = torch.flatten(audio_windows, start_dim=1)

    presence_output = presence_model(audio_windows)
    notes_present = presence_output > 0
    print('Total snaps: {}'.format(len(presence_output)))

    colour_output = colour_model(audio_windows, torch.unsqueeze(notes_present, dim=0))
    notes_blue = colour_output > 0
    finisher_output = finisher_model(audio_windows, torch.unsqueeze(notes_present, dim=0))
    notes_finisher = finisher_output > 0

    for field in fields:
        set_values([(field, fields[field])])
    if 'Title' not in fields:
        _, audio_filename = os.path.split(audio_filepath)
        set_values([('Title', os.path.splitext(audio_filename)[0])])
    if osu_filename is None:
        artist = get_metadata_param()['Artist']
        title = get_metadata_param()['Title']
        creator = get_metadata_param()['Creator']
        version = get_metadata_param()['Version']
        osu_filename = f"{artist} - {title} ({creator}) [{version}].osu"

    with open(osu_filename, 'w') as osu_file:
        osu_file.write('osu file format v14\n\n')

        # General
        osu_file.write('[General]\n')
        for key, value in get_general_param().items():
            osu_file.write('{}: {}\n'.format(key, value))
        osu_file.write('\n')

        # Editor
        osu_file.write('[Editor]\n')
        for key, value in get_editor_param().items():
            osu_file.write('{}: {}\n'.format(key, value))
        osu_file.write('\n')

        # Metadata
        osu_file.write('[Metadata]\n')
        for key, value in get_metadata_param().items():
            osu_file.write('{}: {}\n'.format(key, value))
        osu_file.write('\n')

        # Difficulty
        osu_file.write('[Difficulty]\n')
        for key, value in get_difficulty_param().items():
            osu_file.write('{}: {}\n'.format(key, value))
        osu_file.write('\n')

        # Events
        osu_file.write('[Events]\n')
        osu_file.write('{}\n'.format(get_event_values()[0]))
        osu_file.write('\n')

        # TimingPoints
        osu_file.write('[TimingPoints]\n')
        set_values([
            ('tp_time', str(offset)),
            ('tp_beatLength', str(60000/bpm))
        ])
        tp_params = get_timing_point_param()
        osu_file.write('{},{},{},{},{},{},{},{}\n'.format(
            offset, bar_len, tp_params[2], tp_params[3], tp_params[4], tp_params[5], tp_params[6], tp_params[7]))
        osu_file.write('\n')

        # HitObjects
        osu_file.write('[HitObjects]\n')
        ho_params = get_hit_objects_param()
        for snap_num in range(len(notes_present)):
            if notes_present[snap_num] == False:
                continue
            time_in_ms = snap_to_ms(bar_len, offset, snap_num)
            note_type = (1 + notes_blue[snap_num] + 2 * notes_finisher[snap_num]).item()
            hitsound_number = index_to_hitsound[note_type]
            osu_file.write('{},{},{},{},{},{}\n'.format(
                ho_params[0], ho_params[1], time_in_ms, ho_params[2], hitsound_number, ho_params[3]))

    print(f"File written to {osu_filename}.")

if __name__ == "__main__":
    import torch

    # To replace the deprecated noteFinisherRNN model
    class noFinisher(torch.nn.Module):
        def forward(self, audio_windows, notes_data):
            out = torch.zeros_like(notes_data)
            out = torch.squeeze(out, dim=0)
            return out

    from rnn import notePresenceRNN, noteColourRNN, noteFinisherRNN
    print("Loading state dicts...")
    presence_model = notePresenceRNN()
    colour_model = noteColourRNN()
    # finisher_model = noteFinisherRNN()
    finisher_model = noFinisher()
    if torch.cuda.is_available():
        presence_model = presence_model.cuda()
        presence_model.load_state_dict(torch.load('...', map_location=torch.device('cuda'))) # Change me!
        colour_model = colour_model.cuda()
        colour_model.load_state_dict(torch.load('...', map_location=torch.device('cuda'))) # Change me!
        finisher_model = colour_model.cuda()
        finisher_model.load_state_dict(torch.load('...', map_location=torch.device('cuda'))) # Change me!
    else:
        presence_model.load_state_dict(
            torch.load('...', map_location=torch.device('cpu'))) # Change me!
        colour_model.load_state_dict(
            torch.load('...', map_location=torch.device('cpu'))) # Change me!
        # finisher_model.load_state_dict(
        #     torch.load('...', map_location=torch.device('cpu'))) # Change me!
    print("Model parameters loaded.")
    fields = {}

    audio_filepath = "..." # Change me!
    BPM = 200 # Change me!
    offset = 0 # Change me!

    create_osu_file(presence_model, colour_model, finisher_model, 
            audio_filepath, BPM, offset, osu_filename=None, fields=fields)