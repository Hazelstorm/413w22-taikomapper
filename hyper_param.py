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

# Audio Preprocessing
snap = 4
window_size = 32
max_ms = 180000 # 3 minutes

# Model parameters
notePresenceRNN_embedding_size = 256
notePresenceRNN_RNN_hidden_size = 256
noteColourRNN_embedding_size = 256 # Also applies to noteFinisherRNN
noteColourRNN_hidden_size = 256 # Also applies to noteFinisherRNN