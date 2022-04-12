import numpy as np
import os
from build_to_file_helper import *
from preprocessing_helpers_2 import get_map_audio
import torch
from rnn import taikoRNN


def build_to_file_osu(output, filename, audiofilename = None, title = None):
    
    reverse_onehot_array = np.argmax(output.detach().numpy(), axis=1)
    print('time units: {}'.format(len(reverse_onehot_array)))
    ind_nonzero_array = np.nonzero(reverse_onehot_array)[0]
    print('nonzero time units: {}'.format(len(ind_nonzero_array)))
    
    # set values
    set_values([
        ('AudioFilename', '{}.mp3'.format(audiofilename)),
        ('Title', '{}'.format(title)),
        ('tp_time', '{}'.format(str(ind_nonzero_array[0])))
        # format ('VARTOCHANGE', '{}.mp3'.format(VAR))
    ])

    newfilepath = 'gen_maps\\{}.osu'.format(filename)

    try:
        os.remove(newfilepath)
    except OSError:
        pass

    with open(newfilepath, 'a') as new_osu_file:
        new_osu_file.write('osu file format v14\n\n')

        # General
        new_osu_file.write('[General]\n')
        for key, value in get_general_param():
            new_osu_file.write('{}: {}\n'.format(key, value))
        new_osu_file.write('\n')

        # Editor
        new_osu_file.write('[Editor]\n')
        for key, value in get_editor_param():
            new_osu_file.write('{}: {}\n'.format(key, value))
        new_osu_file.write('\n')

        # Metadata
        new_osu_file.write('[Metadata]\n')
        for key, value in get_metadata_param():
            new_osu_file.write('{}: {}\n'.format(key, value))
        new_osu_file.write('\n')

        # Difficulty
        new_osu_file.write('[Difficulty]\n')
        for key, value in get_difficulty_param():
            new_osu_file.write('{}: {}\n'.format(key, value))
        new_osu_file.write('\n')

        # Events
        new_osu_file.write('[Events]\n')
        new_osu_file.write('{}\n'.format(get_event_values()[0]))
        new_osu_file.write('\n')

        # TimingPoints
        new_osu_file.write('[TimingPoints]\n')
        tp_params = get_timing_point_param()
        new_osu_file.write('{},{},{},{},{},{},{},{}\n'.format(tp_params[0], tp_params[1], tp_params[2], tp_params[3], tp_params[4], tp_params[5], tp_params[6], tp_params[7]))
        new_osu_file.write('\n')

        # HitObjects
        new_osu_file.write('[HitObjects]\n')
        ho_params = get_hit_objects_param()
        for i in ind_nonzero_array:
            new_osu_file.write('{},{},{},{},{},{},{}\n'.format(ho_params[0], ho_params[1], str(i), ho_params[2], str(reverse_onehot_array[i]), ho_params[3], ho_params[4]))

# filepath = 'data\\2021\\203283 Mitchie M - Birthday Song for Miku\\19 Birthday Song for .mp3'
# model = taikoRNN()
# model.load_state_dict(torch.load("Z:\\Users\\David\\Documents\\@@@@UTM\\2022 winter\\CSC 413\\p\\checkpoint\\iter-10000.pt"))
# x = torch.from_numpy(get_map_audio(filepath))
# build_to_file_osu(model(x), '19 Birthday Song for')