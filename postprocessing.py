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
Given a model <model>, an audio file <audio_filepath>, and its bpm and offset, create a .osu file
by using the model to generate a Taiko map.
Parameters:
osu_filename: File name of the generated .osu file. If none, automatically choose a filename.
fields: an optional dictionary containing entries {field_name: value}, to set the field names in postprocessing_helpers.py
        such as "Title".
"""
def create_osu_file(model, audio_filepath, bpm, offset, osu_filename=None, fields={}):
    # See https://osu.ppy.sh/wiki/en/Client/File_formats/Osu_%28file_format%29
    bar_len = 60000/bpm

    spectro = get_map_audio(audio_filepath)
    spectro = torch.tensor(spectro)
    audio_windows = get_audio_around_snaps(spectro, bar_len, offset, hyper_param.window_size)
    audio_windows = torch.flatten(audio_windows, start_dim=1)
    model_output = model(audio_windows)
    notes_present = model_output > 0
    print('Total snaps: {}'.format(len(model_output)))

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
            # hitsound_number = index_to_hitsound[model_output_filtered[snap_num]]
            hitsound_number = 0
            osu_file.write('{},{},{},{},{},{}\n'.format(
                ho_params[0], ho_params[1], time_in_ms, ho_params[2], hitsound_number, ho_params[3]))

import torch
from rnn import notePresenceRNN
presence_model = notePresenceRNN()
if torch.cuda.is_available():
    presence_model = presence_model.cuda()
    presence_model.load_state_dict(torch.load('...'))

else:
    presence_model.load_state_dict(torch.load('ckpt-working-best-trywd-0.00001-with-noise-epoch-220-lr-1e-05.pk', 
        map_location=torch.device('cpu')))

fields = {}

create_osu_file(presence_model, "vajuranda.mp3", 190, 1244, osu_filename=None, fields=fields)
