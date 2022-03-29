import math

"""
All the hyperparameters will be stored here.
"""

# melspectrogram config
n_fft = 512
win_length = None  # default = n_fft
sr = 22000 # Audio Sampling Rate
hop_length = sr // 1000
n_mels = 40
fmin = 0.0
fmax = 6000
power_spectrogram = 2

# hyperparameter config
snap = 12
window_size = 32

# max_snap config
max_len_bpm = 150
max_len_ms = 120000 # 2 minutes

# max_snap calculation
max_len_bar = 60000 / max_len_bpm
max_snap = math.floor(max_len_ms / (max_len_bar / snap))

def get_snap():
    return snap

def get_window_size():
    return window_size

def get_max_snap():
    return max_snap


"""
Returns all parameters for mel.
"""
def get_mel_param():
    return {
        'n_fft': n_fft,
        'win_length': win_length,
        'sr': sr,
        'hop_length': hop_length,
        'n_mels': n_mels,
        'fmin': fmin,
        'fmax': fmax,
        'power_spectrogram': power_spectrogram
    }
