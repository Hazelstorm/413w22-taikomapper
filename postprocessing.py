import numpy as np
import os
from postprocessing_helpers import *
from preprocessing_helpers import get_map_audio
from helper import snap_to_ms, filter_model_output

index_to_hitsound = { \
  1: 0, # don
  2: 2, # kat
  3: 4, # don finisher
  4: 6, # kat finisher
}

def create_osu_file(model, audio_filepath, osu_filename, bpm, offset, title=""):
    # See https://osu.ppy.sh/wiki/en/Client/File_formats/Osu_%28file_format%29
    bar_len = 60000/bpm

    spectro = get_map_audio(audio_filepath)
    model_output = model(torch.unsqueeze(torch.Tensor(spectro), dim=0))
    model_output = torch.squeeze(model_output, dim=0)
    model_output_filtered = filter_model_output(model_output, bar_len, offset, unsnap_tolerance=0) # probability vector
    model_output_filtered = np.argmax(model_output_filtered.detach().numpy(), axis=1) # array of indices from 0-4
    _, audio_filename = os.path.split(audio_filepath)
    print('Total snaps: {}'.format(len(model_output_filtered)))
    
    # set values
    set_values([
        ('AudioFilename', '{}.mp3'.format(audio_filename)),
        ('Title', title),
        ('tp_time', str(offset)),
        ('tp_beatLength', str(60000/bpm))
        # format ('VARTOCHANGE', '{}.mp3'.format(VAR))
    ])

    with open(osu_filename, 'w') as osu_file:
        osu_file.write('osu file format v14\n\n')

        # General
        osu_file.write('[General]\n')
        for key, value in get_general_param():
            osu_file.write('{}: {}\n'.format(key, value))
        osu_file.write('\n')

        # Editor
        osu_file.write('[Editor]\n')
        for key, value in get_editor_param():
            osu_file.write('{}: {}\n'.format(key, value))
        osu_file.write('\n')

        # Metadata
        osu_file.write('[Metadata]\n')
        for key, value in get_metadata_param():
            osu_file.write('{}: {}\n'.format(key, value))
        osu_file.write('\n')

        # Difficulty
        osu_file.write('[Difficulty]\n')
        for key, value in get_difficulty_param():
            osu_file.write('{}: {}\n'.format(key, value))
        osu_file.write('\n')

        # Events
        osu_file.write('[Events]\n')
        osu_file.write('{}\n'.format(get_event_values()[0]))
        osu_file.write('\n')

        # TimingPoints
        osu_file.write('[TimingPoints]\n')
        tp_params = get_timing_point_param()
        osu_file.write('{},{},{},{},{},{},{},{}\n'.format(
            tp_params[0], tp_params[1], tp_params[2], tp_params[3], tp_params[4], tp_params[5], tp_params[6], tp_params[7]))
        osu_file.write('\n')

        # HitObjects
        osu_file.write('[HitObjects]\n')
        ho_params = get_hit_objects_param()
        for snap_num in range(len(model_output_filtered)):
            time_in_ms = snap_to_ms(bar_len, offset, snap_num)
            hitsound_number = index_to_hitsound[model_output_filtered[snap_num]]
            osu_file.write('{},{},{},{},{},{}\n'.format(
                ho_params[0], ho_params[1], time_in_ms, ho_params[2], hitsound_number, ho_params[3]))

import torch
from rnn import taikoRNN
model = taikoRNN()
model.load_state_dict(torch.load('ckpt-1000.pk', map_location=torch.device('cpu')))
create_osu_file(model, "audio.mp3", "test.osu", 200, 12, "Can't Hide Your Love")
