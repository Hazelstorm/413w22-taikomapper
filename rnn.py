import torch
from torch import nn
import hyper_param
from helper import filter_to_snaps

WINDOW_LENGTH = 2 * hyper_param.window_size + 1

# Modified from RNN notebook
class notePresenceRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 50
        self.rnn = nn.GRU(input_size=hyper_param.n_mels, hidden_size=self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, 1)
    
    def forward(self, spectro, bar_len, offset):
        out, _ = self.rnn(spectro)
        out = self.fc(out)
        out = torch.squeeze(out, dim=2)
        out = torch.squeeze(out, dim=0)
        out = filter_to_snaps(out, bar_len, offset, 0)
        return out

class noteColourRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 50
        self.rnn = nn.GRU(input_size=hyper_param.n_mels *  WINDOW_LENGTH, hidden_size=self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, 1)
    
    def forward(self, spectro, notes_data):
        spectro = torch.flatten(spectro, start_dim=2)
        augmented_spectro = torch.cat(spectro, notes_data, 2)
        out, _ = self.rnn(augmented_spectro)
        out = self.fc(out)
        # Allow the model to only colour notes that are present in notes_data
        notes_present = torch.ge(notes_data, torch.zeros_like(notes_data))
        out = torch.mul(out, notes_present)
        return out

class noteFinisherRNN(noteColourRNN):
    def __init__(self):
        super().__init__()
