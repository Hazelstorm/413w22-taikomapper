import math

"""
All the hyperparameters will be stored here.
"""
# melspectrogram config
n_fft = 4096
win_length = None  # default = n_fft
sr = 22000 # Audio Sampling Rate
hop_length = sr // 1000
n_mels = 40
fmin = 20
fmax = 5000
power_spectrogram = 2

# hyperparameter config
snap = 4
window_size = 32

max_ms = 180000 # 3 minutes

def get_snap():
    return snap

def get_window_size():
    return window_size


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
