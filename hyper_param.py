# melspectrogram config
n_fft = 512
win_length = None  # default = n_fft
hop_length = 256  # default = win_length / 4
n_mels = 40
fmin = 0.0
fmax = 6000
power_spectrogram = 2

# snap sampling
snap = 4

def get_snap():
    return snap

def get_mel_param():
    return{
        'n_fft': n_fft,
        'win_length': win_length,
        'hop_length': hop_length,
        'n_mels': n_mels,
        'fmin': fmin,
        'fmax': fmax,
        'power_spectrogram': power_spectrogram
    }